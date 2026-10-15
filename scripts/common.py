"""Shared helpers for the Improved-DDPM demo experiments.

Everything runs on top of the official code (third_party/improved-diffusion) and the
official checkpoints released with the paper (Nichol & Dhariwal, 2021).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

DEMO = Path(__file__).resolve().parent.parent
REPO = DEMO / "third_party" / "improved-diffusion"
CKPT = DEMO / "checkpoints"
DATA = DEMO / "data"
RESULTS = DEMO / "results"
sys.path.insert(0, str(REPO))

from improved_diffusion import gaussian_diffusion as gd  # noqa: E402
from improved_diffusion.respace import SpacedDiffusion, space_timesteps  # noqa: E402
from improved_diffusion.script_util import create_model, model_and_diffusion_defaults  # noqa: E402

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

# Flags copied from the README of openai/improved-diffusion ("Models and Hyperparameters").
MODELS = {
    "cifar_hybrid": dict(
        file="cifar10_uncond_50M_500K.pt",
        model=dict(image_size=32, num_channels=128, num_res_blocks=3, learn_sigma=True, dropout=0.3),
        diffusion=dict(steps=4000, noise_schedule="cosine", rescale_timesteps=True),
    ),
    "cifar_vlb": dict(
        file="cifar10_uncond_vlb_50M_500K.pt",
        model=dict(image_size=32, num_channels=128, num_res_blocks=3, learn_sigma=True, dropout=0.3),
        diffusion=dict(steps=4000, noise_schedule="cosine", rescale_timesteps=True),
    ),
    "imagenet64_hybrid": dict(
        file="imagenet64_uncond_100M_1500K.pt",
        model=dict(image_size=64, num_channels=128, num_res_blocks=3, learn_sigma=True),
        diffusion=dict(steps=4000, noise_schedule="cosine", rescale_timesteps=True),
    ),
}


def load_model(name):
    spec = MODELS[name]
    kw = model_and_diffusion_defaults()
    model_keys = [
        "image_size", "num_channels", "num_res_blocks", "learn_sigma", "class_cond", "use_checkpoint",
        "attention_resolutions", "num_heads", "num_heads_upsample", "use_scale_shift_norm", "dropout",
    ]
    margs = {k: kw[k] for k in model_keys}
    margs.update(spec["model"])
    model = create_model(**margs)
    state = th.load(CKPT / spec["file"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


class MeanOnly(th.nn.Module):
    """Drop the learned-variance channels so the same network can be sampled with a FIXED variance.

    This is an *inference-time ablation* on one checkpoint (same mean network, different Sigma),
    not the paper's separately trained L_simple models.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, **kw):
        return self.model(x, t, **kw)[:, : x.shape[1]]


def make_diffusion(name, use_timesteps=None, K=None, var="learned", ddim_stride=False):
    """var: 'learned' (paper, Eq. 15), 'fixed_large' (sigma^2 = beta_t), 'fixed_small' (sigma^2 = beta~_t)."""
    d = MODELS[name]["diffusion"]
    T = d["steps"]
    betas = gd.get_named_beta_schedule(d["noise_schedule"], T)
    if use_timesteps is None:
        if K is None:
            K = T
        use_timesteps = space_timesteps(T, f"ddim{K}" if ddim_stride else [K])
    var_type = {
        "learned": gd.ModelVarType.LEARNED_RANGE,
        "fixed_large": gd.ModelVarType.FIXED_LARGE,
        "fixed_small": gd.ModelVarType.FIXED_SMALL,
    }[var]
    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=var_type,
        loss_type=gd.LossType.RESCALED_MSE,
        rescale_timesteps=d["rescale_timesteps"],
    )


def to_uint8(x):
    """[-1,1] NCHW tensor -> uint8 NHWC numpy"""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()


def from_uint8(arr):
    """uint8 NHWC numpy -> [-1,1] NCHW float tensor"""
    return th.from_numpy(arr).permute(0, 3, 1, 2).float().div(127.5).sub(1)


def load_npy(name):
    return np.load(DATA / f"{name}.npy")


def save_grid(arr, path, nrow=8, pad=2, scale=1):
    from PIL import Image

    n, h, w, c = arr.shape
    ncol = int(np.ceil(n / nrow))
    grid = np.full((ncol * (h + pad) + pad, nrow * (w + pad) + pad, 3), 16, dtype=np.uint8)
    for i in range(n):
        r, cc = divmod(i, nrow)
        grid[pad + r * (h + pad): pad + r * (h + pad) + h, pad + cc * (w + pad): pad + cc * (w + pad) + w] = arr[i]
    img = Image.fromarray(grid)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)


class Timer:
    def __enter__(self):
        if DEVICE.type == "cuda":
            th.cuda.synchronize()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        if DEVICE.type == "cuda":
            th.cuda.synchronize()
        self.dt = time.perf_counter() - self.t


def seeded_noise(n, shape, seed):
    g = th.Generator(device="cpu").manual_seed(seed)
    return th.randn((n, *shape), generator=g).to(DEVICE)
