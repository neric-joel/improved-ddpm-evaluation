"""Measure per-step GPU latency of the official CIFAR-10 / ImageNet-64 models at several batch sizes."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch as th
from common import load_model, DEVICE, Timer, RESULTS

out = {"device": th.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"}
for name, size in [("cifar_hybrid", 32), ("imagenet64_hybrid", 64)]:
    model = load_model(name)
    nparams = sum(p.numel() for p in model.parameters())
    res = {"params_M": round(nparams / 1e6, 1)}
    for bs in [1, 16, 64, 128] if size == 32 else [1, 8, 32]:
        x = th.randn(bs, 3, size, size, device=DEVICE)
        t = th.full((bs,), 500.0, device=DEVICE)
        with th.no_grad():
            for _ in range(3):
                model(x, t)
            with Timer() as tm:
                for _ in range(10):
                    model(x, t)
        res[f"bs{bs}_ms_per_step"] = round(tm.dt / 10 * 1000, 2)
        print(name, bs, res[f"bs{bs}_ms_per_step"], "ms/step", flush=True)
    out[name] = res
    del model
    th.cuda.empty_cache()
(RESULTS / "benchmark.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
