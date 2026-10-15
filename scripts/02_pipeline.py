"""Experiment pipeline for the Improved-DDPM presentation demo.

Stages (run in this order by default; each stage skips work whose output already exists):
  quick     - CIFAR-10 samples, K in {10,25,50,100,250}, learned vs fixed variances vs DDIM (N=36, shared seeds)
  traj      - reverse-process trajectories (x_t and predicted x_0 snapshots) for the live/explainer view
  latency   - end-to-end batch-size-1 latency vs number of sampling steps K (warm, synchronized)
  ood       - "train on one, test on another": per-image bits/dim (variational bound) on CIFAR-10 test,
              CIFAR-10 train, SVHN test, flipped CIFAR, noisy CIFAR; AUROC
  imagenet  - ImageNet-64 samples at a few K
  nn        - nearest training neighbours of generated samples (memorization check)
  vlb       - full T=4000 VLB terms per timestep (reproduces paper Fig. 2; feeds importance-sampling viz)
  long      - K=1000 and K=4000 samples (learned) to show 4000 vs 50 steps
  fid       - FID + precision/recall vs K for each sampler (exploratory sample size)

Usage:  python -I 02_pipeline.py [stage ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

from common import (  # noqa: E402
    DEVICE, RESULTS, MeanOnly, Timer, from_uint8, load_model, load_npy, make_diffusion, save_grid,
    seeded_noise, space_timesteps, to_uint8,
)

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
th.backends.cudnn.benchmark = True

KS = [10, 25, 50, 100, 250]
SAMPLERS = ["learned", "fixed_large", "fixed_small", "ddim"]
SAMPLES = RESULTS / "samples"
LOG = RESULTS / "pipeline_log.txt"


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


_models = {}


def get_model(name):
    if name not in _models:
        _models.clear()
        th.cuda.empty_cache()
        _models[name] = load_model(name)
    return _models[name]


def sample(name, sampler, K, n, seed0=0, bs=125, fp16=True, image_size=32, progress_every=None):
    """Generate n samples with batch-wise seeds (seed0 + batch index) so every sampler sees the same x_T."""
    model = get_model(name)
    var = "learned" if sampler == "ddim" else sampler
    diff = make_diffusion(name, K=K, var=var, ddim_stride=(sampler == "ddim"))
    net = model if var == "learned" else MeanOnly(model)
    out = []
    for b in range(0, n, bs):
        m = min(bs, n - b)
        seed = seed0 + b // bs
        noise = seeded_noise(m, (3, image_size, image_size), 10_000 + seed)
        th.manual_seed(seed)
        with th.no_grad(), th.autocast("cuda", dtype=th.float16, enabled=fp16):
            if sampler == "ddim":
                x = diff.ddim_sample_loop(net, (m, 3, image_size, image_size), noise=noise, clip_denoised=True, eta=0.0)
            else:
                x = diff.p_sample_loop(net, (m, 3, image_size, image_size), noise=noise, clip_denoised=True)
        out.append(to_uint8(x.float()))
        if progress_every:
            log(f"   {name} {sampler} K={K}: {b + m}/{n}")
    return np.concatenate(out)


# ---------------------------------------------------------------- stages
def stage_quick(n=36):
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for K in KS:
        for s in SAMPLERS:
            path = SAMPLES / f"cifar_{s}_K{K}.npy"
            if path.exists():
                continue
            with Timer() as tm:
                arr = sample("cifar_hybrid", s, K, n, bs=n)
            np.save(path, arr)
            save_grid(arr, SAMPLES / f"cifar_{s}_K{K}.png", nrow=6, scale=3)
            log(f"quick cifar {s} K={K} n={n} {tm.dt:.1f}s")


def stage_long(n=36):
    for s, K in [("learned", 1000), ("learned", 4000), ("fixed_large", 1000), ("fixed_small", 1000), ("ddim", 1000)]:
        path = SAMPLES / f"cifar_{s}_K{K}.npy"
        if path.exists():
            continue
        with Timer() as tm:
            arr = sample("cifar_hybrid", s, K, n, bs=n)
        np.save(path, arr)
        save_grid(arr, SAMPLES / f"cifar_{s}_K{K}.png", nrow=6, scale=3)
        log(f"long cifar {s} K={K} n={n} {tm.dt:.1f}s")


def stage_traj(n=6, snapshots=10):
    """Record x_t and the network's predicted x_0 along the reverse process (learned variance)."""
    out_dir = RESULTS / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, size, Ks in [("cifar_hybrid", 32, [25, 50, 250]), ("imagenet64_hybrid", 64, [50, 250])]:
        for K in Ks:
            for var in ["learned", "fixed_large"]:
                path = out_dir / f"{name}_{var}_K{K}.npz"
                if path.exists():
                    continue
                model = get_model(name)
                diff = make_diffusion(name, K=K, var=var)
                net = model if var == "learned" else MeanOnly(model)
                noise = seeded_noise(n, (3, size, size), 777)
                th.manual_seed(777)
                keep = set(np.linspace(0, K - 1, snapshots).round().astype(int).tolist())
                xs, x0s, ts = [], [], []
                with th.no_grad(), th.autocast("cuda", dtype=th.float16):
                    for i, o in enumerate(diff.p_sample_loop_progressive(net, (n, 3, size, size), noise=noise, clip_denoised=True)):
                        if i in keep:
                            xs.append(to_uint8(o["sample"].float()))
                            x0s.append(to_uint8(o["pred_xstart"].float()))
                            ts.append(int(diff.timestep_map[K - 1 - i]))
                np.savez_compressed(path, x=np.stack(xs), x0=np.stack(x0s), t=np.array(ts), noise=to_uint8(noise.float()))
                log(f"traj {name} {var} K={K}")


def stage_latency():
    path = RESULTS / "latency_bs1.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    model = get_model("cifar_hybrid")
    for K in [10, 25, 50, 100, 250, 1000, 4000]:
        key = f"K{K}"
        if key in res:
            continue
        diff = make_diffusion("cifar_hybrid", K=K, var="learned")
        noise = seeded_noise(1, (3, 32, 32), 1)
        with th.no_grad():
            diff.p_sample_loop(model, (1, 3, 32, 32), noise=noise)  # warm-up
            reps = 3 if K <= 250 else 1
            with Timer() as tm:
                for _ in range(reps):
                    diff.p_sample_loop(model, (1, 3, 32, 32), noise=noise)
        res[key] = round(tm.dt / reps, 3)
        log(f"latency bs1 K={K}: {res[key]} s/image")
        path.write_text(json.dumps(res, indent=2))
    # GAN-like single forward pass reference (one network evaluation)
    with th.no_grad():
        x = th.randn(1, 3, 32, 32, device=DEVICE)
        t = th.zeros(1, device=DEVICE)
        for _ in range(5):
            model(x, t)
        with Timer() as tm:
            for _ in range(20):
                model(x, t)
    res["one_forward_pass"] = round(tm.dt / 20, 4)
    res["device"] = th.cuda.get_device_name(0)
    path.write_text(json.dumps(res, indent=2))


def eval_timesteps(T=4000, K=100):
    """Appendix E: strided subset of K steps plus every t in [0, T/K) to keep the bound tight."""
    return set(space_timesteps(T, [K])) | set(range(T // K))


def bpd_for(name, arr, bs=125, seed=0):
    model = get_model(name)
    diff = make_diffusion(name, use_timesteps=eval_timesteps(), var="learned")
    out = []
    for b in range(0, len(arr), bs):
        x = from_uint8(arr[b: b + bs]).to(DEVICE)
        th.manual_seed(seed + b)
        r = diff.calc_bpd_loop(model, x, clip_denoised=True)
        out.append(r["total_bpd"].cpu().numpy())
    return np.concatenate(out)


def auroc(neg, pos):
    """P(score_pos > score_neg) via rank sums (continuous scores, ties not expected)."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    allv = np.concatenate([neg, pos])
    ranks = allv.argsort().argsort().astype(float) + 1
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stage_ood(n=500, n_small=200):
    out_dir = RESULTS / "ood"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cifar_test, cifar_train, svhn = load_npy("cifar_test"), load_npy("cifar_train"), load_npy("svhn_test")
    sets = {
        "cifar10_test": cifar_test[rng.choice(len(cifar_test), n, replace=False)],
        "svhn_test": svhn[rng.choice(len(svhn), n, replace=False)],
        "cifar10_train": cifar_train[rng.choice(len(cifar_train), n, replace=False)],
    }
    base = cifar_test[rng.choice(len(cifar_test), n_small, replace=False)]
    sets["cifar10_vflip"] = base[:, ::-1].copy()
    noisy = np.clip(base.astype(np.float32) + rng.normal(0, 0.08 * 255, base.shape), 0, 255).astype(np.uint8)
    sets["cifar10_gaussnoise"] = noisy
    for name in ["cifar_hybrid", "cifar_vlb"]:
        for k, arr in sets.items():
            path = out_dir / f"{name}__{k}.npy"
            if path.exists():
                continue
            with Timer() as tm:
                bpd = bpd_for(name, arr)
            np.save(path, bpd)
            if k in ("cifar10_test", "svhn_test"):
                np.save(out_dir / f"examples__{k}.npy", arr[:16])
            log(f"ood {name} {k}: mean {bpd.mean():.3f} bpd (n={len(bpd)}, {tm.dt:.0f}s)")
    # summary
    summ = {}
    for name in ["cifar_hybrid", "cifar_vlb"]:
        d = {}
        for k in sets:
            p = out_dir / f"{name}__{k}.npy"
            if p.exists():
                v = np.load(p)
                d[k] = dict(mean=float(v.mean()), std=float(v.std()), median=float(np.median(v)), n=int(len(v)))
        ref = out_dir / f"{name}__cifar10_test.npy"
        if ref.exists():
            r = np.load(ref)
            for k in sets:
                p = out_dir / f"{name}__{k}.npy"
                if k != "cifar10_test" and p.exists():
                    d[k]["auroc_vs_cifar10_test"] = auroc(r, np.load(p))  # positive = OOD, score = higher bpd
        summ[name] = d
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    log("ood summary", json.dumps(summ))


def stage_imagenet(n=16):
    out = RESULTS / "imagenet"
    out.mkdir(parents=True, exist_ok=True)
    for s, K in [("learned", 25), ("learned", 50), ("learned", 100), ("learned", 250), ("fixed_large", 25), ("fixed_large", 50), ("fixed_small", 25), ("ddim", 25)]:
        path = out / f"imagenet64_{s}_K{K}.npy"
        if path.exists():
            continue
        with Timer() as tm:
            arr = sample("imagenet64_hybrid", s, K, n, bs=n, image_size=64)
        np.save(path, arr)
        save_grid(arr, out / f"imagenet64_{s}_K{K}.png", nrow=8, scale=2)
        log(f"imagenet {s} K={K} {tm.dt:.1f}s")


def stage_nn(k=1):
    """Nearest CIFAR-10 training images (pixel L2) for generated samples."""
    out = RESULTS / "nn"
    out.mkdir(parents=True, exist_ok=True)
    src = SAMPLES / "cifar_learned_K250.npy"
    if not src.exists() or (out / "nn_pixel.npz").exists():
        return
    gen = th.from_numpy(np.load(src)).float().reshape(len(np.load(src)), -1).to(DEVICE) / 255
    train = load_npy("cifar_train")
    tr = th.from_numpy(train).float().reshape(len(train), -1) / 255
    best_d = th.full((len(gen),), 1e9, device=DEVICE)
    best_i = th.zeros(len(gen), dtype=th.long, device=DEVICE)
    for b in range(0, len(tr), 5000):
        chunk = tr[b: b + 5000].to(DEVICE)
        d = th.cdist(gen, chunk)
        v, i = d.min(dim=1)
        upd = v < best_d
        best_d[upd] = v[upd]
        best_i[upd] = i[upd] + b
    # reference: distance from held-out TEST images to their nearest TRAIN image (what "not copied" looks like)
    test = load_npy("cifar_test")[:200]
    te = th.from_numpy(test).float().reshape(len(test), -1).to(DEVICE) / 255
    ref = th.full((len(te),), 1e9, device=DEVICE)
    for b in range(0, len(tr), 5000):
        ref = th.minimum(ref, th.cdist(te, tr[b: b + 5000].to(DEVICE)).min(dim=1).values)
    np.savez_compressed(out / "nn_pixel.npz", gen=np.load(src), nn=train[best_i.cpu().numpy()],
                        dist=best_d.cpu().numpy(), idx=best_i.cpu().numpy(), test_ref_dist=ref.cpu().numpy())
    log(f"nn: gen->train L2 median {best_d.median():.2f}; test->train median {ref.median():.2f}")


def stage_vlb(models=("cifar_hybrid", "cifar_vlb"), n=32):
    out = RESULTS / "vlb_terms"
    out.mkdir(parents=True, exist_ok=True)
    x = load_npy("cifar_test")[:n]
    for name in models:
        path = out / f"{name}.npz"
        if path.exists():
            continue
        model = get_model(name)
        diff = make_diffusion(name, K=4000, var="learned")
        th.manual_seed(0)
        with Timer() as tm:
            r = diff.calc_bpd_loop(model, from_uint8(x).to(DEVICE), clip_denoised=True)
        # calc_bpd_loop iterates t = T-1 ... 0, so column 0 is t = T-1: flip to ascending t
        vb = r["vb"].cpu().numpy()[:, ::-1]  # [n, T] in bits/dim, index 0 = t=0
        np.savez_compressed(path, mean=vb.mean(0), msq=(vb ** 2).mean(0), prior=r["prior_bpd"].cpu().numpy(),
                            total=r["total_bpd"].cpu().numpy(), mse=r["mse"].cpu().numpy()[:, ::-1].mean(0), order="t_ascending")
        log(f"vlb {name}: total {r['total_bpd'].mean():.3f} bpd over {n} imgs ({tm.dt:.0f}s)")


# ---------------- FID / precision / recall ----------------
_incep = None


def inception():
    global _incep
    if _incep is None:
        from pytorch_fid.inception import InceptionV3
        _incep = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEVICE).eval()
    return _incep


def feats(arr, bs=200):
    net = inception()
    out = []
    with th.no_grad():
        for b in range(0, len(arr), bs):
            x = th.from_numpy(arr[b: b + bs]).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
            out.append(net(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def fid_from(mu1, s1, mu2, s2):
    """Frechet distance ||mu1-mu2||^2 + Tr(S1) + Tr(S2) - 2 Tr((S1 S2)^1/2).
    Tr((S1 S2)^1/2) = sum sqrt(eig(A S2 A)) with A = S1^1/2 (both symmetric PSD), which is exact and avoids
    scipy.linalg.sqrtm (pytorch-fid passes a `disp` argument that current SciPy no longer accepts)."""
    w, V = np.linalg.eigh(s1)
    A = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    ev = np.linalg.eigvalsh(A @ s2 @ A)
    tr_covmean = np.sqrt(np.clip(ev, 0, None)).sum()
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def prec_recall(real, fake, k=5):
    r = th.from_numpy(real).float().to(DEVICE)
    f = th.from_numpy(fake).float().to(DEVICE)
    rr = th.cdist(r, r).kthvalue(k + 1, dim=1).values  # radius = distance to k-th NN (excluding self)
    ff = th.cdist(f, f).kthvalue(k + 1, dim=1).values
    d = th.cdist(f, r)  # [nf, nr]
    precision = (d <= rr[None, :]).any(dim=1).float().mean().item()
    recall = (d.t() <= ff[None, :]).any(dim=1).float().mean().item()
    return precision, recall


def stage_fid(n=1000):
    out = RESULTS / "fid"
    out.mkdir(parents=True, exist_ok=True)
    stats = out / "cifar_train_inception.npz"
    if not stats.exists():
        train = load_npy("cifar_train")
        with Timer() as tm:
            ft = feats(train)
        rng = np.random.default_rng(0)
        np.savez_compressed(stats, mu=ft.mean(0), sigma=np.cov(ft, rowvar=False), subset=ft[rng.choice(len(ft), 5000, replace=False)].astype(np.float32))
        log(f"fid: train stats over {len(train)} imgs ({tm.dt:.0f}s)")
    st = np.load(stats)
    mu_r, sig_r, real_sub = st["mu"], st["sigma"], st["subset"]
    resf = out / "fid_results.json"
    res = json.loads(resf.read_text()) if resf.exists() else {}
    # estimator floor: real held-out CIFAR-10 test images scored exactly like the samples
    key = f"real_test_n{n}"
    if key not in res:
        fr = feats(load_npy("cifar_test")[:n])
        p, r = prec_recall(real_sub[:n].astype(np.float64), fr, k=5)
        res[key] = dict(fid=fid_from(fr.mean(0), np.cov(fr, rowvar=False), mu_r, sig_r), precision=p, recall=r, n=n)
        resf.write_text(json.dumps(res, indent=2))
        log(f"fid floor (real test, n={n}): {res[key]}")
    for K in KS:
        for s in SAMPLERS:
            key = f"{s}_K{K}_n{n}"
            if key in res:
                continue
            path = SAMPLES / f"fid_cifar_{s}_K{K}_n{n}.npy"
            if path.exists():
                arr = np.load(path)
            else:
                with Timer() as tm:
                    arr = sample("cifar_hybrid", s, K, n, seed0=1000, bs=125)
                np.save(path, arr)
                log(f"fid samples {s} K={K} n={n} ({tm.dt:.0f}s)")
            fg = feats(arr)
            p, r = prec_recall(real_sub[:n].astype(np.float64), fg, k=5)
            res[key] = dict(fid=fid_from(fg.mean(0), np.cov(fg, rowvar=False), mu_r, sig_r), precision=p, recall=r, n=n, sampler=s, K=K)
            resf.write_text(json.dumps(res, indent=2))
            log(f"fid {s} K={K}: {res[key]}")


STAGES = {
    "quick": stage_quick, "traj": stage_traj, "latency": stage_latency, "ood": stage_ood, "imagenet": stage_imagenet,
    "nn": stage_nn, "vlb": stage_vlb, "long": stage_long, "fid": stage_fid,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for s in args.stages:
        log(f"=== stage {s} ===")
        t0 = time.time()
        STAGES[s]()
        log(f"=== stage {s} done in {time.time() - t0:.0f}s ===")
