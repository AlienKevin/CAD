import argparse
import json
import os
from typing import Dict, Tuple

import torch
from torch.utils.flop_counter import FlopCounterMode

from cad.models.networks.rin import RINClassCond


def _device() -> torch.device:
    # For stable counting, keep model on CPU
    return torch.device("cpu")


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

    c = int(params["num_input_channels"])
    h = int(params["data_size"]) 
    w = int(params["data_size"]) 
    x = torch.randn(batch_size, c, h, w, device=device)

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

    return model, batch


def estimate_flops_with_pytorch(params: Dict[str, int], batch_size: int = 1) -> Dict[str, float]:
    device = _device()
    model, batch = build_model_and_inputs(params, batch_size=batch_size)

    flop_counter = FlopCounterMode(mods=model, display=True, depth=None)
    with flop_counter:
        model(batch)

    total_flops = float(flop_counter.get_total_flops())
    params_count = sum(p.numel() for p in model.parameters())
    return {
        "total": total_flops / 1e9,
        "params_millions": params_count / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate FLOPs with PyTorch FlopCounterMode for RIN presets")
    parser.add_argument(
        "--preset",
        type=str,
        default="cifar10",
        help=(
            "Preset name: CIFAR10, IN-64, IN-64-ABLATION, IN-128, IN-256, IN-512, IN-1024, K-600"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    params = get_preset_params(args.preset)
    result = estimate_flops_with_pytorch(params, batch_size=args.batch_size)

    title = args.preset.upper()
    print(f"FLOPs estimate per sample ({title}) [GFLOPs via PyTorch FlopCounterMode]:")
    print(f"  params [M]: {result['params_millions']:.3f}")
    print(f"Total FLOPs [G]: {result['total']:.3f}")


if __name__ == "__main__":
    main()


