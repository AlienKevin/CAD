import os
import argparse
import numpy as np
from PIL import Image

# Path to the saved samples npz
npz_path = "/svl/u/kevin02/CAD/cad/checkpoints/ImageNet_64_RIN_hf_h200/samples_50k.npz"

# Sampling setup (must match how the samples were generated)
num_fid_samples = 50000
num_classes = 1000
world_size = 8
per_proc_batch_size = 625

# Load images
arr = np.load(npz_path)["arr_0"]  # (num_fid_samples, H, W, 3) uint8
assert arr.shape[0] == num_fid_samples, "npz does not contain expected number of samples"

# Reconstruct index -> class mapping used during saving
n = per_proc_batch_size
global_batch_size = n * world_size
assert num_fid_samples % global_batch_size == 0, "num_fid_samples must be divisible by global batch size"
samples_needed_this_gpu = num_fid_samples // world_size
assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by per-GPU batch size"
iterations = samples_needed_this_gpu // n

# Global balanced class list, then sliced contiguously per-rank as in sampling
all_classes_global = list(range(num_classes)) * (num_fid_samples // num_classes)
subset_len = len(all_classes_global) // world_size

labels = np.empty(num_fid_samples, dtype=np.int32)
for rank in range(world_size):
    local_classes = np.array(
        all_classes_global[rank * subset_len : (rank + 1) * subset_len], dtype=np.int64
    )
    for t in range(iterations):
        for i in range(n):
            idx = t * global_batch_size + i * world_size + rank
            labels[idx] = int(local_classes[t * n + i])

def build_collage(images: np.ndarray, rows: int = 3, cols: int = 3) -> np.ndarray:
    h, w = images.shape[1], images.shape[2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            k = row * cols + col
            grid[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = images[k]
    return grid


def main():
    default_classes = sorted([0, 555, 812, 207, 527, 487, 429, 416, 412, 364, 338, 312, 267, 250, 220, 135])
    parser = argparse.ArgumentParser(description="Create 3x3 collages for given class indices.")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        required=False,
        default=default_classes,
        help="List of class indices (e.g., --classes 0 5 10). Defaults to a preset sorted list.",
    )
    args = parser.parse_args()

    # Sort classes to ensure deterministic order
    args.classes = sorted(args.classes)

    # Validate class indices
    for c in args.classes:
        assert 0 <= c < num_classes, f"Class index {c} out of range [0, {num_classes - 1}]"

    # Output directory
    out_dir = "/svl/u/kevin02/CAD/samples/ImageNet_64_RIN_hf_h200"
    os.makedirs(out_dir, exist_ok=True)

    for c in args.classes:
        sel = np.where(labels == c)[0][:9]
        assert sel.size == 9, f"Did not find 9 samples for class {c}"
        collage = build_collage(arr[sel], rows=3, cols=3)
        out_path = os.path.join(out_dir, f"{c}.png")
        Image.fromarray(collage).save(out_path)
        print(f"Saved collage for class {c} to {out_path} (indices={sel.tolist()})")


if __name__ == "__main__":
    main()
