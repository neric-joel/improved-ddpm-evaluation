"""Probe images for the 'what does the model find likely?' strip: bits/dim of simple synthetic images vs real ones,
scored by the official CIFAR-10 L_hybrid checkpoint with the same evaluation schedule as the OOD experiment."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

from common import DEVICE, RESULTS, from_uint8, load_model, load_npy, make_diffusion, space_timesteps  # noqa: E402


def eval_timesteps(T=4000, K=100):
    return set(space_timesteps(T, [K])) | set(range(T // K))


def probes():
    rng = np.random.default_rng(1)
    H = W = 32
    cif, svhn = load_npy("cifar_test"), load_npy("svhn_test")
    yy, xx = np.mgrid[0:H, 0:W]
    grad = np.stack([xx * 8, yy * 8, 255 - xx * 8], -1).clip(0, 255)
    checker = (((xx // 4 + yy // 4) % 2) * 255)[..., None].repeat(3, -1)
    noisy = np.clip(cif[3].astype(float) + rng.normal(0, 60, cif[3].shape), 0, 255)
    items = [
        ("plain grey", np.full((H, W, 3), 128)),
        ("plain black", np.zeros((H, W, 3))),
        ("smooth gradient", grad),
        ("SVHN digit", svhn[7]),
        ("CIFAR-10 photo", cif[3]),
        ("CIFAR-10 photo (2)", cif[12]),
        ("photo + heavy noise", noisy),
        ("checker board", checker),
        ("random pixels", rng.integers(0, 256, (H, W, 3))),
    ]
    return [(n, a.astype(np.uint8)) for n, a in items]


if __name__ == "__main__":
    model = load_model("cifar_hybrid")
    diff = make_diffusion("cifar_hybrid", use_timesteps=eval_timesteps(), var="learned")
    items = probes()
    x = from_uint8(np.stack([a for _, a in items])).to(DEVICE)
    th.manual_seed(0)
    bpd = diff.calc_bpd_loop(model, x, clip_denoised=True)["total_bpd"].cpu().numpy()
    out = [dict(name=n, bpd=float(b)) for (n, _), b in zip(items, bpd)]
    np.save(RESULTS / "ood" / "probe_images.npy", np.stack([a for _, a in items]))
    (RESULTS / "ood" / "probes.json").write_text(json.dumps(out, indent=2))
    for o in sorted(out, key=lambda d: d["bpd"]):
        print(f"{o['bpd']:7.3f}  {o['name']}")
