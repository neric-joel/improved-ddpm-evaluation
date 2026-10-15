"""Live demo server: serves the presentation app and streams real reverse-diffusion frames from the
official Improved-DDPM checkpoints (Server-Sent Events).

Run from the repository root:   python app/server.py      then open http://127.0.0.1:8766/
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(APP_DIR), "scripts"))

import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from PIL import Image  # noqa: E402

from common import DEVICE, MeanOnly, load_model, make_diffusion, seeded_noise, space_timesteps, to_uint8  # noqa: E402

MODEL_KEYS = {"cifar": ("cifar_hybrid", 32), "imagenet64": ("imagenet64_hybrid", 64)}
_models = {}
_lock = threading.Lock()


def get_model(key):
    name, size = MODEL_KEYS[key]
    if name not in _models:
        _models[name] = load_model(name)
    return _models[name], size


def grid_png(arr, ncol):
    n, h, w, _ = arr.shape
    rows = int(np.ceil(n / ncol))
    out = np.full((rows * (h + 1) - 1, ncol * (w + 1) - 1, 3), 12, np.uint8)
    for i in range(n):
        r, c = divmod(i, ncol)
        out[r * (h + 1): r * (h + 1) + h, c * (w + 1): c * (w + 1) + w] = arr[i]
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_frames(model_key="cifar", K=50, var="learned", seed=0, n=4, max_frames=60):
    """Yield dict frames along the reverse process. Shared by the live server and the offline recorder."""
    model, size = get_model(model_key)
    diff = make_diffusion(MODEL_KEYS[model_key][0], K=K, var=var)
    net = model if var == "learned" else MeanOnly(model)
    noise = seeded_noise(n, (3, size, size), 10_000 + seed)
    th.manual_seed(seed)
    ncol = 2 if n == 4 else min(n, 4)
    every = max(1, int(np.ceil(K / max_frames)))
    t0 = time.perf_counter()
    yield dict(i=-1, K=K, t=int(diff.timestep_map[-1]) + 1, x=grid_png(to_uint8(noise.float()), ncol), x0=None, elapsed=0.0)
    with th.no_grad(), th.autocast("cuda", dtype=th.float16, enabled=DEVICE.type == "cuda"):
        for i, out in enumerate(diff.p_sample_loop_progressive(net, (n, 3, size, size), noise=noise, clip_denoised=True)):
            if i % every == 0 or i == K - 1:
                if DEVICE.type == "cuda":
                    th.cuda.synchronize()
                yield dict(i=i, K=K, t=int(diff.timestep_map[K - 1 - i]), x=grid_png(to_uint8(out["sample"].float()), ncol),
                           x0=grid_png(to_uint8(out["pred_xstart"].float()), ncol), elapsed=round(time.perf_counter() - t0, 3))


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Improved DDPM live demo")

    @app.middleware("http")
    async def no_cache(request, call_next):
        # never let the browser show a stale copy of the demo during a presentation
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/api/health")
    def health():
        return JSONResponse(dict(ok=True, device=th.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu", loaded=list(_models)))

    @app.get("/api/generate")
    def generate(model: str = "cifar", K: int = 50, var: str = "learned", seed: int = 0, n: int = 4):
        if model not in MODEL_KEYS or var not in ("learned", "fixed_large", "fixed_small") or not (2 <= K <= 4000) or not (1 <= n <= 16):
            raise HTTPException(400, "bad parameters")
        if not _lock.acquire(blocking=False):
            raise HTTPException(409, "busy: another generation is running")

        def stream():
            try:
                for fr in generate_frames(model, K, var, seed, n):
                    yield f"data: {json.dumps(fr)}\n\n"
                yield "event: done\ndata: {}\n\n"
            finally:
                _lock.release()

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/bpd")
    def bpd(payload: dict):
        """Score an uploaded image (data URL) with the CIFAR-10 model: variational bound in bits/dim."""
        src = str(payload.get("image", ""))
        if not src.startswith("data:image/") or len(src) > 8_000_000:
            raise HTTPException(400, "expected an image data URL")
        try:
            img = Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1]))).convert("RGB")
        except Exception:
            raise HTTPException(400, "could not decode image")
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((32, 32), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.uint8)[None]
        if not _lock.acquire(blocking=False):
            raise HTTPException(409, "busy")
        try:
            model, _ = get_model("cifar")
            diff = make_diffusion("cifar_hybrid", use_timesteps=set(space_timesteps(4000, [100])) | set(range(40)), var="learned")
            th.manual_seed(0)
            t0 = time.perf_counter()
            x = th.from_numpy(arr).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEVICE)
            val = float(diff.calc_bpd_loop(model, x, clip_denoised=True)["total_bpd"][0])
            return JSONResponse(dict(bpd=val, seconds=round(time.perf_counter() - t0, 2), thumb=grid_png(arr, 1)))
        finally:
            _lock.release()

    app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="app")
    return app


if __name__ == "__main__":
    import uvicorn

    print("Loading CIFAR-10 checkpoint on", DEVICE)
    get_model("cifar")
    # warm-up so the first live click is fast
    for _ in generate_frames("cifar", K=5, n=4):
        pass
    print("Ready: open http://127.0.0.1:8766/")
    uvicorn.run(create_app(), host="127.0.0.1", port=8766, log_level="warning")
