"""Export pipeline results into the data file consumed by the demo app.

Writes:
  app/assets/data/demo_data.js   (window.DEMO_DATA)
  ../03_Visualizer/assets/data/model_data.js   (window.MODEL_DATA), only if that folder exists
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import DEMO, RESULTS  # noqa: E402

ROOT = DEMO.parent
VIS_OUT = ROOT / "03_Visualizer" / "assets" / "data" / "model_data.js"
APP_OUT = DEMO / "app" / "assets" / "data" / "demo_data.js"


def png_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def webp_uri(arr, q=88):
    """Noisy intermediate frames compress badly as PNG; lossy WebP keeps the look at a fraction of the size."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="WEBP", quality=q, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def grid(arr, nrow, pad=1, bg=12):
    n, h, w, _ = arr.shape
    rows = int(np.ceil(n / nrow))
    out = np.full((rows * (h + pad) - pad, nrow * (w + pad) - pad, 3), bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, nrow)
        out[r * (h + pad): r * (h + pad) + h, c * (w + pad): c * (w + pad) + w] = arr[i]
    return out


def film(stack, pad=1, bg=12):
    """stack: [S, n, h, w, 3] -> image with n rows and S columns"""
    S, n, h, w, _ = stack.shape
    out = np.full((n * (h + pad) - pad, S * (w + pad) - pad, 3), bg, np.uint8)
    for s in range(S):
        for i in range(n):
            out[i * (h + pad): i * (h + pad) + h, s * (w + pad): s * (w + pad) + w] = stack[s, i]
    return out


def auroc(neg, pos):
    """P(score_pos > score_neg) with score = bits/dim (higher = flagged as OOD); 0.5 = chance. Rank-sum form (no ties expected)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    allv = np.concatenate([neg, pos])
    ranks = allv.argsort().argsort().astype(float) + 1
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def r4(a):
    return [float(f"{x:.4g}") for x in np.asarray(a).tolist()]


def build():
    data = {"generated_by": "scripts/03_export_data.py"}
    # ---- trajectories
    traj = {}
    tdir = RESULTS / "trajectories"
    nice = {"cifar_hybrid": "CIFAR-10", "imagenet64_hybrid": "ImageNet 64"}
    if tdir.exists():
        for f in sorted(tdir.glob("*.npz")):
            name, rest = f.stem.split("_learned_") if "_learned_" in f.stem else f.stem.split("_fixed_large_")
            var = "learned" if "_learned_" in f.stem else "fixed_large"
            K = int(rest[1:])
            z = np.load(f)
            n_show = 4 if "imagenet" in name else 6
            traj[f.stem] = dict(
                label=f"{nice.get(name, name)} · K={K}" + ("" if var == "learned" else " · fixed β"),
                x=webp_uri(film(z["x"][:, :n_show])), x0=webp_uri(film(z["x0"][:, :n_show])), t=z["t"].tolist(),
                frames=[webp_uri(z["x"][s, 0]) for s in range(z["x"].shape[0])],
                frames_x0=[webp_uri(z["x0"][s, 0]) for s in range(z["x0"].shape[0])],
                model=name, var=var, K=K,
            )
    # order: learned first
    data["traj"] = dict(sorted(traj.items(), key=lambda kv: (kv[1]["var"] != "learned", kv[1]["model"] != "cifar_hybrid", kv[1]["K"])))
    # ---- presets for the forward explorer (generated ImageNet-64 samples: no dataset images redistributed)
    presets = []
    tf = tdir / "imagenet64_hybrid_learned_K250.npz"
    if tf.exists():
        last = np.load(tf)["x"][-1]
        for i in range(min(3, len(last))):
            presets.append(dict(key="default" if i == 0 else f"in64_{i}", label=f"model sample {i + 1}", src=png_uri(last[i])))
    data["presets"] = presets
    # ---- samples per K
    samples = {}
    sdir = RESULTS / "samples"
    for f in sorted(sdir.glob("cifar_*_K*.npy")) if sdir.exists() else []:
        stem = f.stem[len("cifar_"):]
        sampler, K = stem.rsplit("_K", 1)
        arr = np.load(f)[:16]
        samples.setdefault(int(K), {})[sampler] = png_uri(grid(arr, 4))
    data["samples"] = {k: samples[k] for k in sorted(samples)}
    # ---- imagenet samples
    imn = {}
    idir = RESULTS / "imagenet"
    for f in sorted(idir.glob("imagenet64_*_K*.npy")) if idir.exists() else []:
        sampler, K = f.stem[len("imagenet64_"):].rsplit("_K", 1)
        imn.setdefault(int(K), {})[sampler] = png_uri(grid(np.load(f)[:8], 4))
    data["imagenet"] = {k: imn[k] for k in sorted(imn)}
    # ---- vlb terms
    vdir = RESULTS / "vlb_terms"
    vlb = {}
    for f in sorted(vdir.glob("*.npz")) if vdir.exists() else []:
        z = np.load(f)
        # files written by the first pipeline run store calc_bpd_loop's order (t = T-1 first): flip those
        flip = "order" not in z.files
        mean, msq = (z["mean"][::-1], z["msq"][::-1]) if flip else (z["mean"], z["msq"])
        vlb[f.stem] = dict(mean=r4(mean), msq=r4(msq), total=float(z["total"].mean()), prior=float(z["prior"].mean()))
    data["vlb"] = vlb
    # ---- latency
    lf = RESULTS / "latency_bs1.json"
    data["latency"] = json.loads(lf.read_text()) if lf.exists() else None
    # ---- fid
    ff = RESULTS / "fid" / "fid_results.json"
    if ff.exists():
        res = json.loads(ff.read_text())
        curves, pr, n = {}, {}, None
        floor = None
        for k, v in res.items():
            if k.startswith("real_test"):
                floor = v["fid"]
                pr["real"] = [v["precision"], v["recall"]]
                continue
            curves.setdefault(v["sampler"], []).append([v["K"], v["fid"]])
            pr.setdefault(v["sampler"], []).append([v["K"], v["precision"], v["recall"]])
            n = v["n"]
        for s in curves:
            curves[s].sort()
            pr[s].sort()
        data["fid"] = dict(curves=curves, pr=pr, floor=floor, n=n)
    else:
        data["fid"] = None
    # ---- OOD
    odir = RESULTS / "ood"
    ood = None
    if (odir / "cifar_hybrid__cifar10_test.npy").exists() and (odir / "cifar_hybrid__svhn_test.npy").exists():
        sets = {}
        for f in odir.glob("cifar_*__*.npy"):
            model, key = f.stem.split("__")
            sets.setdefault(model, {})[key] = np.load(f)
        summ = json.loads((odir / "summary.json").read_text()) if (odir / "summary.json").exists() else {}
        hb = sets["cifar_hybrid"]
        ood = dict(
            cifar=float(hb["cifar10_test"].mean()), svhn=float(hb["svhn_test"].mean()),
            auroc=auroc(hb["cifar10_test"], hb["svhn_test"]),
            auroc_all={m: {k: auroc(d["cifar10_test"], v) for k, v in d.items() if k != "cifar10_test"} for m, d in sets.items() if "cifar10_test" in d},
            values={m: {k: r4(v) for k, v in d.items()} for m, d in sets.items()},
            summary=summ,
        )
        ex = {}
        for k in ["cifar10_test", "svhn_test"]:
            p = odir / f"examples__{k}.npy"
            if p.exists():
                ex[k] = png_uri(grid(np.load(p)[:8], 8))
        ood["examples"] = ex
        pj, pi = odir / "probes.json", odir / "probe_images.npy"
        if pj.exists() and pi.exists():
            imgs = np.load(pi)
            ood["probes"] = [dict(p, img=png_uri(imgs[i])) for i, p in enumerate(json.loads(pj.read_text()))]
    data["ood"] = ood
    # ---- nearest neighbours
    nf = RESULTS / "nn" / "nn_pixel.npz"
    if nf.exists():
        z = np.load(nf)
        order = np.argsort(z["dist"])  # closest first = most suspicious
        pick = order[:6]
        data["nn"] = dict(
            pairs=[dict(gen=png_uri(z["gen"][i]), nn=png_uri(z["nn"][i]), d=float(z["dist"][i])) for i in pick],
            gen_dist=r4(z["dist"]), test_ref_dist=r4(z["test_ref_dist"]),
        )
    else:
        data["nn"] = None
    bm = RESULTS / "benchmark.json"
    data["benchmark"] = json.loads(bm.read_text()) if bm.exists() else None
    return data


if __name__ == "__main__":
    d = build()
    js = "/* generated by scripts/03_export_data.py — do not edit by hand */\n"
    if VIS_OUT.parent.parent.exists():  # optional companion study page, not part of this repository
        VIS_OUT.parent.mkdir(parents=True, exist_ok=True)
        VIS_OUT.write_text(js + "window.MODEL_DATA = " + json.dumps(d, separators=(",", ":")) + ";\n", encoding="utf-8")
    APP_OUT.parent.mkdir(parents=True, exist_ok=True)
    APP_OUT.write_text(js + "window.DEMO_DATA = " + json.dumps(d, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("wrote", APP_OUT, f"{APP_OUT.stat().st_size / 1e6:.2f} MB")
    print("keys:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in d.items() if k != "generated_by"})
