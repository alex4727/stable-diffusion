import argparse, os, random, shutil
import torch
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import torch.multiprocessing as mp
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--fill_imagenet",
        action='store_true',
        help="whether to fill imagenet"
    )
    parser.add_argument(
        "--imagenet_path",
        type=str,
        help="path to imagenet"
    )
    parser.add_argument(
        "--imagenet_out_path",
        type=str,
        help="path for filled imagenet"
    )
    parser.add_argument(
        "--target_images",
        type=int,
        help="number of target images per class"
    )
    parser.add_argument(
        "--force_target",
        action='store_true',
        help="whether to force target images"
    )
    parser.add_argument(
        "--move_real_images",
        action='store_true',
        help='whether to move real images to fake images (required only once)'
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="start index for imagenet (inclusive)"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=999,
        help="end index for imagenet (inclusive)"
    )
    parser.add_argument(
        "--DDP",
        action='store_true',
        help="use DDP"
    )
    parser.add_argument(
        "--DP",
        action='store_true',
        help="use DP"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus"
    )
    return parser.parse_args()

def load_model_from_config(config, ckpt, verbose=False):
    if verbose:
        print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd and verbose:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def peel_model(model):
    if isinstance(model, DP) or isinstance(model, DDP):
        model = model.module
    return model
    
def save_image_grid(all_gpu_samples, opt, grid=True):
    outpath = opt.outdir
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(opt.outdir, exist_ok=True)  
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    for samples in all_gpu_samples:
        for sample in samples:
            sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(sample.astype(np.uint8))
            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1
    if grid == True:
        n_rows = opt.n_rows if opt.n_rows > 0 else opt.n_samples
        grid = torch.stack(all_gpu_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=n_rows)
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8))
        img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
        grid_count += 1

def save_imagenet(all_gpu_samples, cls_idx, output_path):
    count = 0
    for sample in all_gpu_samples:
        sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(sample.astype(np.uint8))
        # img.save(os.path.join(output_path, f"fake_{cls_idx}_{count:04}.png"))
        img.save(os.path.join(output_path, f"fake_{cls_idx}_{count:04}.JPEG"))
        count += 1

def save_imagenet_chunk(all_gpu_samples, cls_idx, output_path):
    all_gpu_samples = 255. * rearrange(all_gpu_samples.cpu().numpy(), 'b c h w -> b h w c')
    np.save(os.path.join(output_path, f"fake_{cls_idx}.npy"), all_gpu_samples.astype(np.uint8))

def prepare_imagenet_stats(opt):
    print("Preparing imagenet stats")
    imagenet_stats = dict()
    real_images = 0
    with open("ImageNet-LT/ImageNet_labels.txt", "r") as f:
        lines = f.read().splitlines()
        cls_ids, cls_idxs, cls_names = zip(*[i.split(" ") for i in lines])
        for cls_id, cls_idx, cls_name in zip(cls_ids, cls_idxs, cls_names):
            sample_dict = dict()
            sample_dict['name'] = cls_name
            sample_dict['id'] = cls_id
            sample_dict['idx'] = cls_idx
            sample_dict['num'] = 0
            sample_dict['prompt'] = f"a photo of a {cls_name.replace('_', ' ')}"
            imagenet_stats[cls_idx] = sample_dict
            os.makedirs(os.path.join(f"{opt.imagenet_out_path}", "train", f"{cls_id}"), exist_ok=True)

    with open("ImageNet-LT/ImageNet_LT_train.txt", "r") as f:
        lines = f.read().splitlines()
        subset_image_list, subset_label_list = zip(*[i.split(" ") for i in lines])
        count = 0
        prev_label = 0
        for img, label in zip(subset_image_list, subset_label_list):
            imagenet_stats[label]['num'] += 1
            real_images += 1
            if prev_label != label:
                count = 0
                prev_label = label
            if opt.move_real_images:
                new_name = f"{img[:img.rfind('/')]}/real_{label}_{count:04}.JPEG"
                shutil.copy(os.path.join(f"{opt.imagenet_path}", img), os.path.join(f"{opt.imagenet_out_path}", new_name))
            count += 1

    return imagenet_stats, real_images

def run(rank, opt):
    torch.cuda.set_device(rank)
    
    if opt.DDP:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=opt.gpus, rank=rank)
    
    seed_everything(opt.seed + rank)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", verbose=(rank==0))

    if opt.DDP:
        model = peel_model(DDP(model, device_ids=[rank]))
    elif opt.DP:
        model = peel_model(DP(model, device_ids=[0, 1]))
    else:
        pass
    
    sampler = PLMSSampler(model, verbose=(rank==0)) if opt.plms else DDIMSampler(model)
    batch_size = opt.n_samples if not opt.DDP else opt.n_samples // opt.gpus

    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    start_code = None
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                per_gpu_samples = None
                for n in range(opt.n_iter):
                    for prompts in data:
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        per_gpu_samples = x_samples_ddim if per_gpu_samples is None else torch.cat([per_gpu_samples, x_samples_ddim], dim=0)

    all_gpu_samples = [torch.zeros_like(per_gpu_samples) for _ in range(opt.gpus)]
    if opt.DDP:
        dist.all_gather(all_gpu_samples, per_gpu_samples)
    else:
        all_gpu_samples =[per_gpu_samples]
    
    if rank == 0: # save samples and grid
        save_image_grid(all_gpu_samples, opt, grid=True)

def fill(rank, opt):
    torch.cuda.set_device(rank)
    
    if opt.DDP:
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=opt.gpus)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", verbose=(rank==0))

    if opt.DDP:
        model = peel_model(DDP(model, device_ids=[rank]))
    elif opt.DP:
        model = peel_model(DP(model, device_ids=[0, 1]))
    else:
        pass
    
    sampler = PLMSSampler(model, verbose=(False)) if opt.plms else DDIMSampler(model)
    batch_size = opt.n_samples if not opt.DDP else opt.n_samples // opt.gpus
    if rank == 0:
        generated_images_counter = 0

    for cls_idx, cls_dict in opt.imagenet_stats.items():
        if int(cls_idx) < int(opt.start_idx):
            continue
        if int(cls_idx) > int(opt.end_idx):
            if opt.DDP:
                dist.barrier()
            break
        if rank == 0:
            current_time = datetime.now().strftime('%Y_%b%d_%H-%M-%S')
            print(f"{current_time} | Start Generating {cls_dict['name']} (id: {cls_idx}, num real: {cls_dict['num']}) | {generated_images_counter}/{opt.total_images_to_generate} generated so far", flush=True)
        num_generate = opt.target_images - cls_dict['num'] if not opt.force_target else opt.target_images
        n_iter = num_generate // opt.n_samples if num_generate % opt.n_samples == 0 else num_generate // opt.n_samples + 1
        n_drop = n_iter * opt.n_samples - num_generate 
        prompt = cls_dict['prompt']
        data = [batch_size * [prompt]]

        start_code = None
        precision_scope = autocast if opt.precision=="autocast" else nullcontext

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    per_gpu_samples = None
                    for n in range(n_iter):
                        for prompts in data:
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=batch_size,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            per_gpu_samples = x_samples_ddim if per_gpu_samples is None else torch.cat([per_gpu_samples, x_samples_ddim], dim=0)

        all_gpu_samples = [torch.zeros_like(per_gpu_samples) for _ in range(opt.gpus)]
        if opt.DDP:
            dist.all_gather(all_gpu_samples, per_gpu_samples)
        else:
            all_gpu_samples =[per_gpu_samples]
        if rank == 0: 
            all_gpu_samples = torch.cat(all_gpu_samples, dim=0)
            all_gpu_samples = all_gpu_samples[:-n_drop] if n_drop > 0 else all_gpu_samples
            generated_images_counter += all_gpu_samples.shape[0]
            save_imagenet_chunk(all_gpu_samples, cls_idx, os.path.join(f"{opt.imagenet_out_path}", "train", f"{cls_dict['id']}"))
        
        if opt.DDP:
            dist.barrier()

def main():
    opt = get_args()
    if opt.fill_imagenet:
        opt.imagenet_stats, real_images = prepare_imagenet_stats(opt)
        opt.total_images_to_generate = opt.target_images*len(opt.imagenet_stats)-real_images if not opt.force_target else opt.target_images*len(opt.imagenet_stats)
        print(f"Total Real images: {real_images}")
        print(f"Total Images to Generate: {opt.total_images_to_generate}")
        print(f"Generating from index {opt.start_idx} to {opt.end_idx}")
        print("Few options like n_iter, prompt will be automatically set to generate images for ImageNet")
    if opt.n_samples % opt.gpus != 0:
        raise ValueError("n_samples must be divisible by gpus")
    if opt.DDP:
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))
            if opt.fill_imagenet:
                mp.spawn(fill, nprocs=opt.gpus, args=(opt,))
            else:
                mp.spawn(run, nprocs=opt.gpus, args=(opt,))
        except KeyboardInterrupt:
            dist.destroy_process_group()
    else:
        if opt.fill_imagenet:
            fill(0, opt)
        else:
            run(0, opt)

if __name__ == "__main__":
    main()