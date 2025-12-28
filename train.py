import torch, os, glob, random, copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from argparse import ArgumentParser
from time import time
from tqdm import tqdm
from omegaconf import OmegaConf
from dataset import RealESRGANDataset, RealESRGANDegrader
from model import Net
from ram.models.ram_lora import ram
from torchvision import transforms
from utils import add_lora_to_unet

dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
world_size = dist.get_world_size()

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--torch_dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="dtype used to load diffusion backbone to reduce VRAM usage")
parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                    help="turn on gradient checkpointing for UNet modules")
parser.add_argument("--enable_attention_slicing", action="store_true",
                    help="enable attention slicing to reduce peak memory in attention layers")
parser.add_argument("--use_xformers", action="store_true",
                    help="enable xFormers memory efficient attention if available")

args = parser.parse_args()

# fixed seed for reproduction
seed = rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

config = OmegaConf.load("config.yml")

epoch = args.epoch
learning_rate = args.learning_rate
bsz = args.batch_size

torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.torch_dtype]

device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

if rank == 0:
    print("batch size per gpu =", bsz)
    print(f"backbone dtype = {torch_dtype}")

from diffusers import StableDiffusionPipeline
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

if args.enable_attention_slicing:
    pipe.enable_attention_slicing()

if args.use_xformers:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional dependency
        if rank == 0:
            print(f"[warn] failed to enable xFormers attention: {exc}")

pipe.to(device)

vae = pipe.vae
tokenizer = pipe.tokenizer
unet = pipe.unet
text_encoder = pipe.text_encoder

unet_D = copy.deepcopy(unet)
new_conv_in = torch.nn.Conv2d(
    256, 320, 3, padding=1, device=device, dtype=unet_D.conv_in.weight.dtype
)
new_conv_in.weight.data = unet_D.conv_in.weight.data.repeat(1, 64, 1, 1) / 64
new_conv_in.bias.data = unet_D.conv_in.bias.data
unet_D.conv_in = new_conv_in
unet_D = add_lora_to_unet(unet_D)
unet_D.set_adapters(["default_encoder", "default_decoder", "default_others"])

if args.enable_gradient_checkpointing:
    unet_D.enable_gradient_checkpointing()
    unet.enable_gradient_checkpointing()
    pipe.text_encoder.gradient_checkpointing_enable()

if args.enable_attention_slicing:
    unet.enable_attention_slicing()
    unet_D.enable_attention_slicing()

vae_teacher = copy.deepcopy(vae)
unet_teacher = copy.deepcopy(unet)

osediff = torch.load("./weight/pretrained/osediff.pkl", weights_only=False)
vae_teacher.load_state_dict(osediff["vae"])
unet_teacher.load_state_dict(osediff["unet"])

vae_teacher.to(device=device, dtype=torch_dtype)
unet_teacher.to(device=device, dtype=torch_dtype)

if args.enable_gradient_checkpointing:
    unet_teacher.enable_gradient_checkpointing()

if args.enable_attention_slicing:
    unet_teacher.enable_attention_slicing()

from diffusers.models.autoencoders.vae import Decoder 
ckpt_halfdecoder = torch.load("./weight/pretrained/halfDecoder.ckpt", weights_only=False)
decoder = Decoder(in_channels=4,
                  out_channels=3,
                  up_block_types=["UpDecoderBlock2D" for _ in range(4)],
                  block_out_channels=[64, 128, 256, 256],
                  layers_per_block=2,
                  norm_num_groups=32,
                  act_fn="silu",
                  norm_type="group",
                  mid_block_add_attention=True)
decoder_ckpt = {}
for k, v in ckpt_halfdecoder["state_dict"].items():
    if "decoder" in k:
        new_k = k.replace("decoder.", "")
        decoder_ckpt[new_k] = v
decoder.load_state_dict(decoder_ckpt, strict=True)
decoder.to(device=device, dtype=torch_dtype)

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DAPE = ram(pretrained="./weight/pretrained/ram_swin_large_14m.pth",
           pretrained_condition="./weight/pretrained/DAPE.pth",
           image_size=384,
           vit="swin_l").eval().to(device=device, dtype=torch_dtype)

vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)
vae_teacher.requires_grad_(False)
unet_teacher.requires_grad_(False)
decoder.requires_grad_(False)
DAPE.requires_grad_(False)

model = DDP(Net(unet, copy.deepcopy(decoder)).to(device), device_ids=[rank])
model_D = DDP(unet_D.to(device), device_ids=[rank])
model.requires_grad_(True)
model_D.requires_grad_(False)
params_to_opt = []
for n, p in model_D.named_parameters():
    if "lora" in n or "conv_in" in n:
        p.requires_grad = True
        params_to_opt.append(p)

if rank == 0:
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt/1e6, "M")

dataset = RealESRGANDataset(config, bsz)
degrader = RealESRGANDegrader(config, device)
dataloader = DataLoader(dataset, batch_size=bsz, num_workers=8)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(params_to_opt, lr=1e-6)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,], gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

model_dir = "./%s" % (args.model_dir,)
log_path = "./%s/log.txt" % (args.log_dir,)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

print("start training...")
timesteps = torch.tensor([999], device=device).long().expand(bsz,)
alpha = pipe.scheduler.alphas_cumprod[999]
for epoch_i in range(1, epoch + 1):
    start_time = time()
    loss_avg = 0.0
    loss_distil_avg = 0.0
    loss_adv_avg = 0.0
    loss_D_avg = 0.0
    iter_num = 0
    dist.barrier()
    for batch in tqdm(dataloader):
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                LR, HR = degrader.degrade(batch)
                text_input = tokenizer(DAPE.generate_tag(ram_transforms(LR))[0],
                                       max_length=tokenizer.model_max_length,
                                       padding="max_length", return_tensors="pt").to(device)
                encoder_hidden_states = text_encoder(text_input.input_ids, return_dict=False)[0]
                LR, HR = LR * 2 - 1, HR * 2 - 1
                LR_ = F.interpolate(LR, scale_factor=4, mode="bicubic")
                LR_latents = vae_teacher.encode(LR_).latent_dist.mean * vae_teacher.config.scaling_factor
                HR_latents = vae.encode(HR).latent_dist.mean
                pred_teacher = unet_teacher(
                    LR_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
                z0_teacher = (LR_latents-((1-alpha)**0.5)*pred_teacher)/(alpha**0.5)
                z0_teacher = vae_teacher.post_quant_conv(z0_teacher / vae_teacher.config.scaling_factor)
                z0_teacher = decoder.conv_in(z0_teacher)
                z0_teacher = decoder.mid_block(z0_teacher)
                z0_gt = vae.post_quant_conv(HR_latents)
                z0_gt = decoder.conv_in(z0_gt)
                z0_gt = decoder.mid_block(z0_gt)
            z0_student = model(LR)
            loss_distil = (z0_student - z0_teacher).abs().mean()
            loss_adv = F.softplus(-model_D(
                z0_student,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]).mean()
            loss = loss_distil + loss_adv
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.cuda.amp.autocast(enabled=True):
            pred_real = model_D(
                z0_gt.detach(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            pred_fake = model_D(
                z0_student.detach(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            loss_D = F.softplus(pred_fake).mean() + F.softplus(-pred_real).mean()
        optimizer_D.zero_grad(set_to_none=True)
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()
        loss_avg += loss.item()
        loss_distil_avg += loss_distil.item()
        loss_adv_avg += loss_adv.item()
        loss_D_avg += loss_D.item()
        iter_num += 1
        # print("loss", loss.item())
        # print("loss_distil", loss_distil.item())
        # print("loss_adv", loss_adv.item())
        # print("loss_D", loss_D.item())
    scheduler.step()
    loss_avg /= iter_num
    loss_distil_avg /= iter_num
    loss_adv_avg /= iter_num
    loss_D_avg /= iter_num
    log_data = "[%d/%d] Average loss: %f, distil loss: %f, adv loss: %f, D loss: %f, time cost: %.2fs, cur lr is %f." % (epoch_i, epoch, loss_avg, loss_distil_avg, loss_adv_avg, loss_D_avg, time() - start_time, scheduler.get_last_lr()[0])
    if rank == 0:
        print(log_data)
        with open(log_path, "a") as log_file:
            log_file.write(log_data + "\n")
        if epoch_i % args.save_interval == 0:
            torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))
