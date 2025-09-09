import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.core.global_hydra import GlobalHydra
from PIL import Image
from tqdm import tqdm

# Ensure project root is on sys.path when Hydra changes cwd
root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)

from cad.models.diffusion import DiffusionModule


def create_npz_from_sample_folder(sample_dir: Path, num: int = 50_000) -> Path:
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(sample_dir / f"{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = sample_dir.with_suffix("")  # samples/{exp_name}
    npz_path = npz_path.with_name(npz_path.name)  # keep same directory name
    npz_path = sample_dir.parent / f"{sample_dir.name}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def is_distributed_available() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except Exception:
        return False


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    # Defaults similar to reference script
    num_fid_samples = 50000
    per_proc_batch_size = 2000
    seed = 42

    # Torch backend perf flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    # Setup DDP if available
    distributed = is_distributed_available() and torch.cuda.is_available()
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        seed = seed + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        if rank == 0:
            print(
                f"Starting rank={rank}, seed={seed}, world_size={world_size}."
            )
    else:
        rank = 0
        world_size = 1
        device = 0 if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        print(f"Starting single process, seed={seed}.")

    # Load model
    ckpt_path = Path(cfg.checkpoints.dirpath) / "last.ckpt"
    print(f'{ckpt_path=}')
    model: DiffusionModule = DiffusionModule.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.model,
        strict=False,
    )
    model = model.to(device)
    model.eval()

    # Derive sampling shape and num classes from config (CIFAR-10 or ImageNet)
    num_input_channels = int(cfg.model.network.num_input_channels)
    data_res = int(cfg.data.data_resolution)
    num_classes = int(cfg.data.label_dim)
    shape = (num_input_channels, data_res, data_res)

    # Output directory: samples/{exp_name}/
    sample_folder_dir = Path(cfg.root_dir) / "samples" / cfg.experiment_name
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    if distributed:
        dist.barrier()

    # Compute iterations and class assignment
    n = per_proc_batch_size
    global_batch_size = n * world_size
    assert (
        num_fid_samples % global_batch_size == 0
    ), "num_fid_samples must be divisible by global batch size"
    if rank == 0:
        print(f"Total number of images that will be sampled: {num_fid_samples}")

    samples_needed_this_gpu = int(num_fid_samples // world_size)
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)

    # Balanced class list
    all_classes = list(range(num_classes)) * (
        num_fid_samples // num_classes
    )
    subset_len = len(all_classes) // world_size
    all_classes = np.array(
        all_classes[rank * subset_len : (rank + 1) * subset_len], dtype=np.int64
    )

    cur_idx = 0
    total = 0
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    for _ in pbar:
        y = torch.from_numpy(all_classes[cur_idx * n : (cur_idx + 1) * n])
        cur_idx += 1

        # One-hot class conditioning (B, num_classes)
        cond = torch.nn.functional.one_hot(y.long(), num_classes=num_classes).float()
        cond = cond.to(model.device)

        with torch.no_grad():
            samples = model.sample(
                batch_size=cond.shape[0],
                shape=shape,
                cond=cond,
                stage="test",
                cfg=cfg.model.cfg_rate if hasattr(cfg.model, "cfg_rate") else 0.0,
            )
            # samples: (B, C, H, W) in uint8 or float; convert to PIL savable arrays
            if samples.dtype != torch.uint8:
                # Map from [-1,1] to [0,255] if needed
                samples = ((samples + 1) / 2.0).clamp(0, 1) * 255.0
            samples = samples.to(torch.uint8).cpu()

        for i in range(samples.shape[0]):
            index = i * world_size + rank + total
            img = samples[i].permute(1, 2, 0).numpy()
            Image.fromarray(img).save(sample_folder_dir / f"{index:06d}.png")
        total += global_batch_size

    if distributed:
        dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        print("Done.")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    # Reset any existing Hydra instance (defensive when running multiple times in a notebook)
    GlobalHydra.instance().clear()
    main()


