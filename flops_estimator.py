import math
import json
import os
from typing import Dict, Tuple


def _flops_linear(num_tokens: int, in_dim: int, out_dim: int) -> int:
    """FLOPs for a dense layer: 2 * N * in * out (multiply-add)."""
    return 2 * num_tokens * in_dim * out_dim


def _flops_conv2d(
    in_h: int,
    in_w: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
) -> int:
    """Estimate FLOPs for a standard conv2d per sample.

    Formula: 2 * out_h * out_w * out_channels * (in_channels * k_h * k_w)
    Bias addition is ignored as it is small compared to MACs.
    """
    out_h = math.floor((in_h + 2 * padding - kernel_size) / stride) + 1
    out_w = math.floor((in_w + 2 * padding - kernel_size) / stride) + 1
    kernel_mults = in_channels * kernel_size * kernel_size
    return 2 * out_h * out_w * out_channels * kernel_mults


def _flops_conv3d(
    in_t: int,
    in_h: int,
    in_w: int,
    in_channels: int,
    out_channels: int,
    k_t: int,
    k_h: int,
    k_w: int,
    s_t: int,
    s_h: int,
    s_w: int,
    padding_t: int = 0,
    padding_h: int = 0,
    padding_w: int = 0,
) -> int:
    """Estimate FLOPs for a standard conv3d per sample.

    Formula: 2 * out_t * out_h * out_w * out_channels * (in_channels * k_t * k_h * k_w)
    Bias addition is ignored.
    """
    out_t = math.floor((in_t + 2 * padding_t - k_t) / s_t) + 1
    out_h = math.floor((in_h + 2 * padding_h - k_h) / s_h) + 1
    out_w = math.floor((in_w + 2 * padding_w - k_w) / s_w) + 1
    kernel_mults = in_channels * k_t * k_h * k_w
    return 2 * out_t * out_h * out_w * out_channels * kernel_mults


def _params_linear(in_dim: int, out_dim: int, use_bias: bool = True) -> int:
    params = in_dim * out_dim
    if use_bias:
        params += out_dim
    return params


def _params_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    use_bias: bool = True,
) -> int:
    params = out_channels * in_channels * kernel_size * kernel_size
    if use_bias:
        params += out_channels
    return params


def _params_conv3d(
    in_channels: int,
    out_channels: int,
    k_t: int,
    k_h: int,
    k_w: int,
    use_bias: bool = True,
) -> int:
    params = out_channels * in_channels * k_t * k_h * k_w
    if use_bias:
        params += out_channels
    return params


def _params_mlp(dim_model: int, hidden_multiplier: int, use_bias: bool = True, gated: bool = False) -> int:
    ff = dim_model * hidden_multiplier
    if not gated:
        return _params_linear(dim_model, ff, use_bias) + _params_linear(ff, dim_model, use_bias)
    # Gated MLP: In, Gate, Out projections
    return (
        _params_linear(dim_model, ff, use_bias)
        + _params_linear(dim_model, ff, use_bias)
        + _params_linear(ff, dim_model, use_bias)
    )


def _params_attention(dim_q: int, dim_kv: int, att_dim: int, dim_out: int, use_bias: bool = True) -> int:
    # Q, K, V and output projection
    return (
        _params_linear(dim_q, att_dim, use_bias)
        + _params_linear(dim_kv, att_dim, use_bias)
        + _params_linear(dim_kv, att_dim, use_bias)
        + _params_linear(att_dim, dim_out, use_bias)
    )


def _flops_attention(
    q_len: int,
    kv_len: int,
    dim_q: int,
    dim_kv: int,
    num_heads: int,
    attention_dim: int = 0,
) -> Tuple[int, Dict[str, int]]:
    """Attention FLOPs following the provided reference formula.

    We assume query, key, value share num_heads and head_dim.
    For cross-attention we allow dim_q != dim_kv. attention_dim is the projected
    multi-head dimension; when 0, we follow the code's default: min(dim_q, dim_kv) for CA,
    dim_q for SA. The caller should pass the value it uses.

    Returns total FLOPs and a breakdown dict.
    """
    att_dim = attention_dim if attention_dim > 0 else min(dim_q, dim_kv)
    # Projections
    flops_q = _flops_linear(q_len, dim_q, att_dim)
    # K and V are computed from "from" tokens (kv_len)
    flops_k = _flops_linear(kv_len, dim_kv, att_dim)
    flops_v = _flops_linear(kv_len, dim_kv, att_dim)

    # Attention logits and weighted sum (Score @ V)
    # 2 * Lq * Lk * (num_heads * head_dim) == 2 * Lq * Lk * att_dim
    flops_logits = 2 * q_len * kv_len * att_dim
    # Softmax: 3 * num_heads * Lq * Lk
    flops_softmax = 3 * num_heads * q_len * kv_len
    # Weighted sum: 2 * Lq * Lk * att_dim
    flops_weighted_sum = 2 * q_len * kv_len * att_dim

    # Output projection back to model dim_q
    flops_out = _flops_linear(q_len, att_dim, dim_q)

    total = (
        flops_q
        + flops_k
        + flops_v
        + flops_logits
        + flops_softmax
        + flops_weighted_sum
        + flops_out
    )
    breakdown = {
        "q_proj": flops_q,
        "k_proj": flops_k,
        "v_proj": flops_v,
        "logits": flops_logits,
        "softmax": flops_softmax,
        "score_times_v": flops_weighted_sum,
        "out_proj": flops_out,
    }
    return total, breakdown


def _flops_mlp(num_tokens: int, dim_model: int, hidden_multiplier: int, gated: bool = False) -> Tuple[int, Dict[str, int]]:
    """FLOPs for MLP.

    - Standard (used in RIN): two projections d -> ff, ff -> d
    - Gated (Transformer++ style, optional): In, Gate, Out projections + elementwise gating
      Reference: 2 * N * (3 * d * ff) for projections, plus 5 * N * d for gating.
    """
    ff = dim_model * hidden_multiplier
    if not gated:
        in_proj = _flops_linear(num_tokens, dim_model, ff)
        out_proj = _flops_linear(num_tokens, ff, dim_model)
        total = in_proj + out_proj
        return total, {"in_proj": in_proj, "out_proj": out_proj}
    else:
        proj_flops = 2 * num_tokens * (3 * dim_model * ff)
        gating_flops = 5 * num_tokens * dim_model
        total = proj_flops + gating_flops
        return total, {"proj": proj_flops, "gating": gating_flops}


def estimate_flops(params: Dict) -> Dict[str, int]:
    """Estimate forward-pass FLOPs for RINBackbone per sample.

    Expected keys in params (with typical defaults):
      - data_size (int): input resolution (height==width)
      - data_dim (int)
      - num_input_channels (int)
      - num_latents (int)
      - latents_dim (int)
      - label_dim (int)
      - num_processing_layers (int)
      - num_blocks (int)
      - path_size (int)
      - num_cond_tokens (int)
      - read_write_heads (int)
      - compute_heads (int)
      - latent_mlp_multiplier (int)
      - data_mlp_multiplier (int)
      - use_cond_token (bool)
      - concat_cond_token_to_latents (bool)
      - use_cond_rin_block (bool)  # if True, additional CA over cond is used

    Optional keys for video / custom tokens:
      - num_frames (int): number of frames for video inputs (default: 1)
      - temporal_patch_size (int): temporal patch size / stride (default: 1)
      - num_patches_override (int): if provided, force token count to this value

    Returns a dictionary with a breakdown and a "total" key.
    """
    # Read params with reasonable defaults
    data_size = int(params.get("data_size", 64))
    data_dim = int(params.get("data_dim", 256))
    num_input_channels = int(params.get("num_input_channels", 3))
    num_latents = int(params.get("num_latents", 128))
    latents_dim = int(params.get("latents_dim", 512))
    label_dim = int(params.get("label_dim", 0))
    num_processing_layers = int(params.get("num_processing_layers", 2))
    num_blocks = int(params.get("num_blocks", 3))
    path_size = int(params.get("path_size", 2))
    num_cond_tokens = int(params.get("num_cond_tokens", 0))
    read_write_heads = int(params.get("read_write_heads", 16))
    compute_heads = int(params.get("compute_heads", 16))
    latent_mlp_multiplier = int(params.get("latent_mlp_multiplier", 4))
    data_mlp_multiplier = int(params.get("data_mlp_multiplier", 4))
    use_cond_token = bool(params.get("use_cond_token", True))
    use_cond_rin_block = bool(params.get("use_cond_rin_block", False))

    # Video parameters (default to image)
    num_frames = int(params.get("num_frames", 1))
    temporal_patch_size = int(params.get("temporal_patch_size", 1))

    # Core lengths
    patches_per_side = data_size // path_size
    if num_frames > 1 or temporal_patch_size > 1:
        time_patches = num_frames // temporal_patch_size
        num_patches = patches_per_side * patches_per_side * time_patches
    else:
        num_patches = patches_per_side * patches_per_side

    if "num_patches_override" in params:
        num_patches = int(params["num_patches_override"])
    
    print(f'{num_patches=} (patch_size={path_size}x{path_size})')

    # In implementation, z length is effectively num_latents for both concat and non-concat modes.
    z_len = num_latents
    cond_len = (1 + num_cond_tokens) if use_cond_token else 1  # time token always present

    breakdown: Dict[str, int] = {}
    # Parameter counting
    total_params = 0

    # 1) Patch extractor conv (2D for images, 3D for video)
    if num_frames > 1 or temporal_patch_size > 1:
        breakdown["patch_conv"] = _flops_conv3d(
            in_t=num_frames,
            in_h=data_size,
            in_w=data_size,
            in_channels=num_input_channels,
            out_channels=data_dim,
            k_t=temporal_patch_size,
            k_h=path_size,
            k_w=path_size,
            s_t=temporal_patch_size,
            s_h=path_size,
            s_w=path_size,
        )
        total_params += _params_conv3d(num_input_channels, data_dim, temporal_patch_size, path_size, path_size)
    else:
        breakdown["patch_conv"] = _flops_conv2d(
            in_h=data_size,
            in_w=data_size,
            in_channels=num_input_channels,
            out_channels=data_dim,
            kernel_size=path_size,
            stride=path_size,
            padding=0,
        )
        total_params += _params_conv2d(num_input_channels, data_dim, path_size)

    # 2) Time embedder (map_time: dim=t_dim -> 4*t_dim -> 4*t_dim)
    # t_dim = latents_dim // 4, per sample, 1 token
    t_dim = latents_dim // 4
    time_in = _flops_linear(1, t_dim, 4 * t_dim)
    time_out = _flops_linear(1, 4 * t_dim, 4 * t_dim)
    breakdown["time_embedder"] = time_in + time_out
    # time embedder params: t_dim->4*t_dim, 4*t_dim->4*t_dim
    total_params += _params_linear(t_dim, 4 * t_dim) + _params_linear(4 * t_dim, 4 * t_dim)

    # 3) Conditioning mapping (approximate): map label/cond tokens to latents_dim
    # For class conditioning this is typically 1 token. For text, pass num_cond_tokens via params.
    if use_cond_token and label_dim > 0:
        breakdown["cond_mapping"] = _flops_linear(cond_len - 1, label_dim, latents_dim)
        total_params += _params_linear(label_dim, latents_dim)
    else:
        breakdown["cond_mapping"] = 0

    # 4) Previous latents MLP on z_len tokens
    prev_mlp, _ = _flops_mlp(z_len, latents_dim, latent_mlp_multiplier, gated=False)
    breakdown["previous_latents_mlp"] = prev_mlp
    total_params += _params_mlp(latents_dim, latent_mlp_multiplier)

    # Helper: attention_dim selections matching code
    # - CrossAttention: attention_dim = min(dim_q, dim_kv)
    # - SelfAttention: attention_dim = dim_qkv

    # 5) RIN blocks
    rin_total = 0
    # Aggregates for detailed breakdown (do not include in total calculation)
    ca_total = 0
    ca_cond_total = 0
    sa_total_all = 0
    mlp_total_all = 0
    for _ in range(num_blocks):
        # Retriever CA: latents attend to data
        ca1, _ = _flops_attention(
            q_len=z_len,
            kv_len=num_patches,
            dim_q=latents_dim,
            dim_kv=data_dim,
            num_heads=read_write_heads,
            attention_dim=min(latents_dim, data_dim),
        )
        total_params += _params_attention(latents_dim, data_dim, min(latents_dim, data_dim), latents_dim)
        mlp1, _ = _flops_mlp(z_len, latents_dim, latent_mlp_multiplier, gated=False)
        total_params += _params_mlp(latents_dim, latent_mlp_multiplier)

        # (Optional) Cond CA inside RINBlockCond: latents attend to cond tokens
        if use_cond_rin_block and cond_len > 0:
            ca_cond, _ = _flops_attention(
                q_len=z_len,
                kv_len=cond_len,
                dim_q=latents_dim,
                dim_kv=latents_dim,
                num_heads=read_write_heads,
                attention_dim=latents_dim,
            )
            total_params += _params_attention(latents_dim, latents_dim, latents_dim, latents_dim)
        else:
            ca_cond = 0

        # Processor SA (num_processing_layers): latents self-attention
        sa_total = 0
        mlp_sa_total = 0
        for _pl in range(num_processing_layers):
            sa, _ = _flops_attention(
                q_len=z_len,
                kv_len=z_len,
                dim_q=latents_dim,
                dim_kv=latents_dim,
                num_heads=compute_heads,
                attention_dim=latents_dim,
            )
            total_params += _params_attention(latents_dim, latents_dim, latents_dim, latents_dim)
            sa_total += sa
            mlp_sa, _ = _flops_mlp(z_len, latents_dim, latent_mlp_multiplier, gated=False)
            mlp_sa_total += mlp_sa
            total_params += _params_mlp(latents_dim, latent_mlp_multiplier)

        # Writer CA: data attends to latents
        ca2, _ = _flops_attention(
            q_len=num_patches,
            kv_len=z_len,
            dim_q=data_dim,
            dim_kv=latents_dim,
            num_heads=read_write_heads,
            attention_dim=min(latents_dim, data_dim),
        )
        total_params += _params_attention(data_dim, latents_dim, min(latents_dim, data_dim), data_dim)
        mlp2, _ = _flops_mlp(num_patches, data_dim, data_mlp_multiplier, gated=False)
        total_params += _params_mlp(data_dim, data_mlp_multiplier)

        rin_total += ca1 + mlp1 + ca_cond + sa_total + mlp_sa_total + ca2 + mlp2
        # Update aggregates
        ca_total += ca1 + ca2
        ca_cond_total += ca_cond
        sa_total_all += sa_total
        mlp_total_all += (mlp1 + mlp_sa_total + mlp2)

    breakdown["rin_blocks"] = rin_total

    # 6) Output mapping: tokens -> patches (linear)
    if num_frames > 1 or temporal_patch_size > 1:
        out_patch_dim = num_input_channels * temporal_patch_size * path_size * path_size
    else:
        out_patch_dim = num_input_channels * path_size * path_size
    breakdown["tokens_to_patches"] = _flops_linear(num_patches, data_dim, out_patch_dim)
    total_params += _params_linear(data_dim, out_patch_dim)

    total = sum(breakdown.values())
    breakdown["total"] = total
    # Add aggregate categories for reporting (not included in total)
    breakdown["cross_attention"] = ca_total
    breakdown["cond_cross_attention"] = ca_cond_total
    breakdown["self_attention"] = sa_total_all
    breakdown["mlp"] = mlp_total_all
    breakdown["params_total"] = total_params
    return breakdown


__all__ = ["estimate_flops"]


if __name__ == "__main__":
    import argparse

    def get_preset_params(name: str) -> Dict[str, int]:
        name = name.upper()
        config_path = os.path.join(os.path.dirname(__file__), "rin_estimator_configs.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)
        presets = cfg.get("presets", {})
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}")
        return presets[name]

    parser = argparse.ArgumentParser(description="Estimate FLOPs for RIN configs")
    parser.add_argument(
        "--preset",
        type=str,
        default="cifar10",
        help=(
            "Preset name: cifar10, IN-64, IN-128, IN-256, IN-512, IN-1024, K-600"
        ),
    )
    args = parser.parse_args()

    params = get_preset_params(args.preset)
    flops_breakdown = estimate_flops(params)
    title = args.preset.upper() if args.preset else "CIFAR-10"
    # Print params first (in millions)
    params_millions = flops_breakdown.get("params_total", 0) / 1e6
    print(f"Params: {params_millions:.3f}M")
    print(f"FLOPs estimate per sample ({title}) [GFLOPs]:")
    for key, value in flops_breakdown.items():
        if key in ("total", "params_total"):
            continue
        print(f"  {key}: {value/1e9:.3f}")
    print(f"Total: {flops_breakdown['total']/1e9:.3f}")
