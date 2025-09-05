import argparse
import json
import os
from typing import Dict, Tuple

import torch
from ptflops import get_model_complexity_info

from cad.models.networks.rin import RINClassCond


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preset_params(name: str) -> Dict[str, int]:
    name = name.upper()
    config_path = os.path.join(os.path.dirname(__file__), "rin_estimator_configs.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    presets = cfg.get("presets", {})
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}")
    return presets[name]


def build_model_and_inputs(params: Dict[str, int], batch_size: int = 1) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    device = _device()
    model = RINClassCond(
        data_size=int(params["data_size"]),
        data_dim=int(params["data_dim"]),
        num_input_channels=int(params["num_input_channels"]),
        num_latents=int(params["num_latents"]),
        latents_dim=int(params["latents_dim"]),
        label_dim=int(params["label_dim"]),
        num_processing_layers=int(params["num_processing_layers"]),
        num_blocks=int(params["num_blocks"]),
        path_size=int(params["path_size"]),
        num_cond_tokens=int(params["num_cond_tokens"]),
        read_write_heads=int(params["read_write_heads"]),
        compute_heads=int(params["compute_heads"]),
        latent_mlp_multiplier=int(params["latent_mlp_multiplier"]),
        data_mlp_multiplier=int(params["data_mlp_multiplier"]),
        use_cond_token=bool(params.get("use_cond_token", True)),
        concat_cond_token_to_latents=bool(params.get("concat_cond_token_to_latents", True)),
        use_cond_rin_block=bool(params.get("use_cond_rin_block", False)),
        retrieve_attention_scores=True,
    ).to(device)
    model.eval()

    c = int(params["num_input_channels"])  # channels
    h = int(params["data_size"])          # height
    w = int(params["data_size"])          # width
    x = torch.randn(batch_size, c, h, w, device=device)

    # gamma (timestep/noise): expected as a 1D vector [B]
    gamma = torch.randn(batch_size, device=device)

    # previous_latents: [B, num_latents, latents_dim]
    prev_latents = torch.randn(
        batch_size, int(params["num_latents"]), int(params["latents_dim"]), device=device
    )

    # label one-hot or embeddings: use dense vector of size label_dim
    label = torch.randn(batch_size, int(params["label_dim"]), device=device)

    batch = {
        "y": x,
        "gamma": gamma,
        "previous_latents": prev_latents,
        "label": label,
    }

    return model, batch


def estimate_flops_with_ptflops(params: Dict[str, int], batch_size: int = 1) -> Dict[str, float]:
    preset_name = params.get("preset_name", "")
    if preset_name in ("K-600", "K600"):
        print("Warning: K-600 video preset isn't fully represented in the 2D backbone; reporting spatial FLOPs only.")

    # Build model on CPU for stable ptflops hooks
    device = torch.device("cpu")
    model = RINClassCond(
        data_size=int(params["data_size"]),
        data_dim=int(params["data_dim"]),
        num_input_channels=int(params["num_input_channels"]),
        num_latents=int(params["num_latents"]),
        latents_dim=int(params["latents_dim"]),
        label_dim=int(params["label_dim"]),
        num_processing_layers=int(params["num_processing_layers"]),
        num_blocks=int(params["num_blocks"]),
        path_size=int(params["path_size"]),
        num_cond_tokens=int(params["num_cond_tokens"]),
        read_write_heads=int(params["read_write_heads"]),
        compute_heads=int(params["compute_heads"]),
        latent_mlp_multiplier=int(params["latent_mlp_multiplier"]),
        data_mlp_multiplier=int(params["data_mlp_multiplier"]),
        use_cond_token=bool(params.get("use_cond_token", True)),
        concat_cond_token_to_latents=bool(params.get("concat_cond_token_to_latents", True)),
        use_cond_rin_block=bool(params.get("use_cond_rin_block", False)),
        retrieve_attention_scores=True,
    ).to(device)
    model.eval()

    c = int(params["num_input_channels"])  # channels
    h = int(params["data_size"])          # height
    w = int(params["data_size"])          # width

    def input_constructor(input_res):
        _c, _h, _w = input_res
        x = torch.randn(batch_size, _c, _h, _w, device=device)
        gamma = torch.randn(batch_size, device=device)
        prev_latents = torch.randn(
            batch_size, int(params["num_latents"]), int(params["latents_dim"]), device=device
        )
        label = torch.randn(batch_size, int(params["label_dim"]), device=device)
        batch = {
            "y": x,
            "gamma": gamma,
            "previous_latents": prev_latents,
            "label": label,
        }
        return (batch,)

    macs, params_count = get_model_complexity_info(
        model,
        (c, h, w),
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True,
        input_constructor=input_constructor,
    )
    flops = macs * 2.0
    result = {
        "total": flops / 1e9,
        "macs": macs / 1e9,
        "params_millions": params_count / 1e6,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Estimate FLOPs with ptflops for RIN presets")
    parser.add_argument(
        "--preset",
        type=str,
        default="cifar10",
        help=(
            "Preset name: cifar10, IN-64, IN-64-ablation IN-128, IN-256, IN-512, IN-1024, K-600"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    params = get_preset_params(args.preset)
    params["preset_name"] = args.preset.upper()
    result = estimate_flops_with_ptflops(params, batch_size=args.batch_size)

    title = args.preset.upper()
    print(f"FLOPs estimate per sample ({title}) [GFLOPs via ptflops]:")
    print(f"  params [M]: {result['params_millions']:.3f}")
    print(f"  MACs [G]: {result['macs']:.3f}")
    print(f"Total FLOPs [G] (2x MACs): {result['total']:.3f}")


if __name__ == "__main__":
    main()


