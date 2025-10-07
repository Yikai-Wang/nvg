import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from pytorch_lightning import seed_everything
from nvg.models.generator import NVGenerator
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Multi-GPU inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="batch size per GPU")
    parser.add_argument("--samples_per_class", default=50, type=int, help="total number of samples to generate")
    parser.add_argument("--eval_ema", action="store_true", help="use ema model")
    parser.add_argument("--eval_content_ema", action='store_true')
    parser.add_argument("--eval_structure_ema", action='store_true')
    parser.add_argument("--sample_dir", default="samples", type=str, help="directory to save samples")
    parser.add_argument("--content_cfg_scale", default=1.0, type=str, help="cfg scale for content")
    parser.add_argument("--structure_cfg_scale", default=1.0, type=str, help="cfg scale for structure")
    parser.add_argument("--structure_sampling_step", default=50, type=int, help="sampling step for structure")
    parser.add_argument("--use_gumbel_topk", action="store_true", help="use gumbel top-k sampling")
    parser.add_argument("--top_k", default='0', type=str, help="top k for sampling")
    parser.add_argument("--temperature", default=1.0, type=float, help="temperature for sampling")
    parser.add_argument("--top_p", default=1.0, type=str, help="top p for sampling")
    parser.add_argument("--save_png", action="store_true", help="save samples as PNG files instead of .npz")
    parser.add_argument("--full_list", action="store_true")
    parser.add_argument("--return_structure", action="store_true")

    # Multi-GPU options
    parser.add_argument("--multi_gpu_mode", default="ddp", choices=["ddp", "dp"],
                       help="Multi-GPU mode: 'ddp' for DistributedDataParallel, 'dp' for DataParallel")
    parser.add_argument("--dist_port", default="12355", type=str, help="port for distributed training")

    return parser.parse_args()


def create_npz_from_samples(samples, sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_generator(config, ckpt_path=None, use_ema=False, use_content_ema=False, use_structure_ema=False, structure_ckpt_path=None, ignore_keys=['first_stage_model', 'repa_encoder']):
    model = NVGenerator(**config.model.params)
    print(f"Loading checkpoint from {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    if use_ema:
        print('******Setting use_ema=TRUE, Will ignore component ema setting******')
        print("Loading EMA generator from checkpoint.")
        keys = list(sd.keys())
        for k in keys:
            if k.startswith("ema_model"):
                sd[k.replace("ema_model.", "")] = sd[k]
                del sd[k]
    elif use_content_ema or use_structure_ema:
        print("USE_CONTENT_EMA:{}; USE_STRUCTURE_EMA:{}".format(use_content_ema, use_structure_ema))
        keys = list(sd.keys())
        for k in keys:
            if k.startswith("ema_model"):
                if 'structure' in k:
                    if use_structure_ema:
                        sd[k.replace("ema_model.", "")] = sd[k]
                    del sd[k]
                elif 'content' in k or 'x0_head' in k or 'cls_head' in k:
                    if use_content_ema:
                        sd[k.replace("ema_model.", "")] = sd[k]
                    del sd[k]
                else:
                    del sd[k]
    else:
        print("Loading generator from checkpoint without EMA.")
        keys = list(sd.keys())
        for k in keys:
            if k.startswith("ema_model"):
                del sd[k]
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                del sd[k]
    sd = {k.split('nvgformer.')[1]: v for k, v in sd.items()}
    missing, unexpected = model.nvgformer.load_state_dict(sd, strict=False)
    print(f"missing keys: {missing} unexpected keys: {unexpected}")
    return model.eval()

def negative_log_sequence(start, end, num, base=np.e):
    """
    Generate a decreasing sequence from `start` to `end` using a negative logarithmic curve.

    Parameters:
        start (float): The starting value (larger).
        end (float): The ending value (smaller).
        num (int): Number of values.
        base (float): Logarithm base, e.g., np.e, 10, 2.

    Returns:
        np.ndarray: A sequence of `num` values decreasing from `start` to `end`.
    """
    # Step 1: Create linearly spaced values in log space
    log_start = 0
    log_end = np.log(num - 1) / np.log(base) if num > 1 else 0
    log_space = np.logspace(log_start, log_end, num=num, base=base)

    # Step 2: Flip and normalize to [0, 1]
    values = -log_space
    values = (values - values.min()) / (values.max() - values.min())

    # Step 3: Scale to [end, start]
    return values * (start - end) + end

def setup_distributed(rank, world_size, port='12355'):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def generate_samples_distributed(rank, world_size, args):
    """Main generation function for distributed processing"""
    # Setup distributed training
    setup_distributed(rank, world_size, args.dist_port)

    # Set seed for reproducibility
    seed_everything(42 * rank)

    # Set device for this process
    device = torch.device(f"cuda:{rank}")

    # Load model
    config = load_config(args.config_file, display=False)
    config.model.params.use_ema = False
    model = load_generator(config,
                           ckpt_path=args.ckpt_path,
                           use_ema=args.eval_ema,
                           use_content_ema=args.eval_content_ema,
                           use_structure_ema=args.eval_structure_ema)
    model = model.to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    # Enable optimizations
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # Calculate samples per GPU
    total_samples = 1000 * args.samples_per_class
    samples_per_gpu = total_samples // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu

    # Adjust for last GPU to handle remainder
    if rank == world_size - 1:
        end_idx = total_samples
        samples_per_gpu = end_idx - start_idx

    if rank == 0:
        print(f"Total samples: {total_samples}")
        print(f"Samples per GPU: {samples_per_gpu}")
    print(f"GPU {rank}: generating samples {start_idx} to {end_idx-1}")

    # Generate labels for this GPU
    labels_full = torch.arange(1000, device=device).repeat(args.samples_per_class)
    labels_this_gpu = labels_full[start_idx:end_idx]

    samples = []
    if args.full_list or args.return_structure:
        grids = []
    num_batches = (samples_per_gpu + args.batch_size - 1) // args.batch_size

    if '-' in args.content_cfg_scale:
        content_cfg = args.content_cfg_scale.split('-')
        content_cfg_start, content_cfg_end = map(float, content_cfg)
        content_cfg_scale = np.linspace(content_cfg_start, content_cfg_end, num=9)
        content_use_cfg = True
    else:
        content_cfg_scale = float(args.content_cfg_scale)
        content_cfg_scale = [content_cfg_scale] * 9
        if content_cfg_scale[0] == 1:
            content_use_cfg = False
        else:
            content_use_cfg = True


    if '-' in args.structure_cfg_scale:
        structure_cfg = args.structure_cfg_scale.split('-')
        structure_cfg_start, structure_cfg_end = map(float, structure_cfg)
        structure_cfg_scale = np.linspace(structure_cfg_start, structure_cfg_end, num=9)
        structure_use_cfg = True
    else:
        structure_cfg_scale = float(args.structure_cfg_scale)
        structure_cfg_scale = [structure_cfg_scale] * 9
        if structure_cfg_scale[0] == 1:
            structure_use_cfg = False
        else:
            structure_use_cfg = True

    if '-' in args.top_k:
        top_k_range = args.top_k.split('-')
        top_k_start, top_k_end = map(int, top_k_range)
        args.top_k = np.linspace(top_k_start, top_k_end, num=9).astype(int)
    else:
        args.top_k = int(args.top_k)
        args.top_k = [args.top_k] * 9

    if '-' in args.top_p:
        top_p_range = args.top_p.split('-')
        top_p_start, top_p_end = map(float, top_p_range)
        args.top_p = negative_log_sequence(top_p_start, top_p_end, num=9)
    else:
        args.top_p = float(args.top_p)
        args.top_p = [args.top_p] * 9

    print(f"Top K values: {args.top_k}")
    print(f"Top P values: {args.top_p}")

    if content_use_cfg:
        print(f"Using content CFG scale: {content_cfg_scale}")
    if structure_use_cfg:
        print(f"Using structure CFG scale: {structure_cfg_scale}")

    with torch.inference_mode():
        pbar = tqdm(range(num_batches), ncols=0, desc=f"GPU {rank}") if rank == 0 else range(num_batches)

        for i in pbar:
            batch_start = i * args.batch_size
            batch_end = min(batch_start + args.batch_size, samples_per_gpu)
            current_batch_size = batch_end - batch_start

            if current_batch_size <= 0:
                break


            class_label = labels_this_gpu[batch_start:batch_end]
            structure_noise = torch.randn((current_batch_size, 256, 8), device=device)

            imgs = model.module.generate_images(
                class_label=class_label,
                structure_noise=structure_noise,
                content_use_cfg=content_use_cfg,
                content_cfg_scale=content_cfg_scale,
                structure_use_cfg=structure_use_cfg,
                structure_cfg_scale=structure_cfg_scale,
                structure_sampling_step=args.structure_sampling_step,
                top_k=args.top_k,
                temperature=args.temperature,
                top_p=args.top_p,
                full_list=args.full_list,
                return_structure=args.return_structure,
                use_gumbel_topk=args.use_gumbel_topk,
            )
            if args.full_list or args.return_structure:
                imgs, grid = imgs

            imgs = (imgs + 1.) / 2.0
            sample = torch.clamp(255 * imgs, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
            samples.append(sample)

            if args.full_list or args.return_structure:
                grid = (grid + 1.) / 2.0
                grid = torch.clamp(255 * grid, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
                grids.append(grid)

    # Concatenate all samples from this GPU
    if samples:
        samples = np.concatenate(samples, axis=0)

        # Save samples from this GPU
        os.makedirs(os.path.dirname(args.sample_dir), exist_ok=True)
        gpu_sample_path = f"{args.sample_dir}_gpu{rank}"
        create_npz_from_samples(samples, gpu_sample_path, num=samples.shape[0])

    if args.full_list or args.return_structure:
        if grids:
            grids = np.concatenate(grids, axis=0)
            grid_path = f"{args.sample_dir}_grid_gpu{rank}.npz"
            np.savez(grid_path, arr_0=grids)
            print(f"Saved grid samples from GPU {rank} to {grid_path}")

    # Wait for all processes to finish
    dist.barrier()

    # Merge all samples on rank 0
    if rank == 0:
        print("Merging samples from all GPUs...")
        all_samples = []

        for gpu_rank in range(world_size):
            gpu_sample_path = f"{args.sample_dir}_gpu{gpu_rank}.npz"
            if os.path.exists(gpu_sample_path):
                gpu_samples = np.load(gpu_sample_path)['arr_0']
                all_samples.append(gpu_samples)
                # Remove individual GPU files
                os.remove(gpu_sample_path)

        if all_samples:
            final_samples = np.concatenate(all_samples, axis=0)
            if args.save_png:
                for i in range(1000):
                    os.makedirs(os.path.join(args.sample_dir, f"{i:04d}"), exist_ok=True)
                for i, sample in enumerate(final_samples):
                    img = Image.fromarray(sample)
                    img.save(os.path.join(args.sample_dir, f"{labels_full[i]:04d}", f"{i:05d}.png"))
            else:
                np.random.shuffle(final_samples)  # Shuffle samples across all GPUs
                create_npz_from_samples(final_samples, args.sample_dir, num=final_samples.shape[0])
                print(f"Successfully merged {final_samples.shape[0]} samples from {world_size} GPUs")

        if args.full_list or args.return_structure:
            all_grids = []
            for gpu_rank in range(world_size):
                grid_path = f"{args.sample_dir}_grid_gpu{gpu_rank}.npz"
                if os.path.exists(grid_path):
                    gpu_grids = np.load(grid_path)['arr_0']
                    all_grids.append(gpu_grids)
                    os.remove(grid_path)

            if all_grids:
                final_grids = np.concatenate(all_grids, axis=0)
                for i in range(1000):
                    os.makedirs(os.path.join(args.sample_dir, f"grid_{i:04d}"), exist_ok=True)
                for i, grid in enumerate(final_grids):
                    img = Image.fromarray(grid)
                    img.save(os.path.join(args.sample_dir, f"grid_{labels_full[i]:04d}", f"{i:05d}.png"))

    cleanup_distributed()


def main():
    args = get_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    num_gpus = torch.cuda.device_count()
    print(f"Using DistributedDataParallel with {num_gpus} GPUs")
    mp.spawn(generate_samples_distributed, args=(num_gpus, args), nprocs=num_gpus, join=True)

if __name__ == "__main__":
    main()