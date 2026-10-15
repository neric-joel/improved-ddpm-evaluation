/* demo.js: presenter story mode for the Improved DDPM live demo. */
(function () {
  "use strict";
  const D = window.DDPM, M = window.DEMO_DATA || {}, R = window.REPLAY || {};
  const $ = (id) => document.getElementById(id);
  const COL = { cyan: "var(--ours)", amber: "var(--base)", coral: "var(--bad)", mint: "var(--good)", violet: "var(--alt)", blue: "var(--alt2)", text: "var(--text)", faint: "var(--faint)", muted: "var(--muted)" };
  const raf = (fn) => { let p = false, a; return function () { a = arguments; if (p) return; p = true; requestAnimationFrame(() => { p = false; fn.apply(null, a); }); }; };
  const segc = (id, onPick) => $(id).addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; setSeg(id, b.dataset.v); onPick(b.dataset.v); });
  const BT = '<span class="bt">β</span>';  // beta-tilde drawn with a positioned tilde (U+0303 is not in the embedded fonts)
  const setSeg = (id, v) => $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.v === String(v)));

  // ------------------------------------------------------------ stage + navigation
  const stage = $("stage");
  const fit = () => { const s = Math.min(innerWidth / 1920, innerHeight / 1080); stage.style.transform = `scale(${s})`; };
  window.addEventListener("resize", fit); fit();

  const scenes = Array.from(document.querySelectorAll(".scene"));
  const hooks = {};
  let cur = 0;
  const rail = $("steps");
  let mainCount = 0;
  scenes.forEach((s, i) => {
    if (!s.dataset.backup) mainCount++;
    const firstBackup = s.dataset.backup && !scenes[i - 1].dataset.backup;
    if (s.dataset.backup && !firstBackup) return;
    const b = document.createElement("button");
    b.innerHTML = s.dataset.backup ? "Q&amp;A" : `<b>${i + 1}</b>${s.dataset.name}`;
    if (s.dataset.backup) b.classList.add("backup");
    b.dataset.idx = i;
    b.onclick = () => go(i);
    rail.appendChild(b);
  });
  const railBtns = Array.from(rail.querySelectorAll("button"));
  function go(i) {
    i = Math.max(0, Math.min(scenes.length - 1, i));
    scenes.forEach((s, j) => { s.classList.toggle("on", j === i); s.classList.toggle("left", j < i); });
    railBtns.forEach((b) => { const j = +b.dataset.idx; const qa = b.classList.contains("backup"); b.classList.toggle("on", qa ? !!scenes[i].dataset.backup : j === i); b.classList.toggle("done", !qa && j < i); });
    cur = i;
    const h = hooks[scenes[i].id];
    if (h && h.enter) h.enter();
    try { history.replaceState(null, "", "#" + scenes[i].id); } catch (e) { /* file:// */ }
  }

  // ------------------------------------------------------------ live server detection
  const live = { ok: false, device: "" };
  (function detect() {
    const pill = $("livePill");
    const set = (ok, text) => { pill.className = "live-pill " + (ok ? "on" : "off"); pill.querySelector("span").textContent = text; const g = $("g-live"); if (g) g.textContent = ok ? "Generate on the GPU" : "Play recorded run"; };
    if (!/^https?:/.test(location.protocol)) { set(false, "Offline: recorded runs"); return; }
    const ctl = new AbortController();
    const to = setTimeout(() => ctl.abort(), 1500);
    fetch("api/health", { signal: ctl.signal }).then((r) => r.json()).then((j) => {
      clearTimeout(to); live.ok = !!j.ok; live.device = j.device;
      set(live.ok, live.ok ? `Live: ${String(j.device).replace("NVIDIA GeForce ", "").replace(" Laptop GPU", " laptop GPU")}` : "Offline: recorded runs");
    }).catch(() => set(false, "Offline: recorded runs"));
  })();

  // ------------------------------------------------------------ timer
  const timer = { el: $("timer"), on: false, start: 0, id: null };
  function toggleTimer() {
    timer.on = !timer.on;
    timer.el.classList.toggle("show", timer.on);
    if (timer.on) {
      timer.start = performance.now();
      timer.id = setInterval(() => {
        const left = 360 - (performance.now() - timer.start) / 1000;
        const a = Math.abs(left), m = Math.floor(a / 60), s = Math.floor(a % 60);
        timer.el.textContent = (left < 0 ? "−" : "") + m + ":" + String(s).padStart(2, "0");
        timer.el.classList.toggle("warn", left < 60 && left >= 0);
        timer.el.classList.toggle("over", left < 0);
      }, 250);
    } else clearInterval(timer.id);
  }

  // ============================================================ 1 · NOISE
  (function sceneNoise() {
    const SIZE = 64, T = 1000;
    const AB = { linear: D.cumprod(D.linearBetas(T)), cosine: D.cumprod(D.cosineBetas(T)) };
    const ab = (s, t) => (t <= 0 ? 1 : AB[s][Math.min(T, t) - 1]);
    const eps = D.gaussians(SIZE * SIZE * 3, 42);
    let x0 = D.imageToArray(D.proceduralImage(SIZE), SIZE);
    let t = 350, playId = null;
    const thr = 0.05 * 0.05;
    const waste = (s) => { let n = 0; for (let k = 1; k <= T; k++) if (ab(s, k) < thr) n++; return n / T; };
    $("n-wl").textContent = Math.round(100 * waste("linear")) + "%";
    $("n-wc").textContent = Math.round(100 * waste("cosine")) + "%";
    const state = (a) => { const s = Math.sqrt(a); return s > 0.93 ? "almost clean" : s > 0.8 ? "shape still visible" : s > 0.6 ? "shape fading" : s > 0.4 ? "hard to recognize" : s > 0.15 ? "barely visible" : "pure noise"; };
    const filmT = [0, 0.25, 0.5, 0.75, 1];
    const mkFilm = (id) => { const r = $(id); r.innerHTML = filmT.map(() => "<canvas></canvas>").join(""); r.insertAdjacentHTML("afterend", `<div class="minifilm-cap">${filmT.map((u) => `<span>${u}</span>`).join("")}</div>`); return Array.from(r.children); };
    const filmL = mkFilm("n-lin-film"), filmC = mkFilm("n-cos-film");
    const drawFilms = () => filmT.forEach((u, i) => { D.drawNoised(filmL[i], x0, eps, ab("linear", Math.round(u * T)), SIZE); D.drawNoised(filmC[i], x0, eps, ab("cosine", Math.round(u * T)), SIZE); });
    const xs = Array.from({ length: 201 }, (_, i) => i / 200);
    const cl = xs.map((u) => [u, ab("linear", Math.round(u * T))]), cc = xs.map((u) => [u, ab("cosine", Math.round(u * T))]);
    const render = raf(() => {
      const al = ab("linear", t), ac = ab("cosine", t);
      D.drawNoised($("n-lin"), x0, eps, al, SIZE);
      D.drawNoised($("n-cos"), x0, eps, ac, SIZE);
      $("n-lin-state").textContent = state(al); $("n-cos-state").textContent = state(ac);
      $("n-lin-a").textContent = `ᾱ = ${al.toFixed(3)}`; $("n-cos-a").textContent = `ᾱ = ${ac.toFixed(3)}`;
      $("n-t").textContent = `t/T = ${(t / T).toFixed(2)}`;
      $("n-slider").value = t;
      const u = t / T;
      const near = filmT.reduce((b, v, i) => (Math.abs(v - u) < Math.abs(filmT[b] - u) ? i : b), 0);
      filmL.forEach((c, i) => c.classList.toggle("hot", i === near)); filmC.forEach((c, i) => c.classList.toggle("hot", i === near));
      D.chart($("n-chart"), {
        width: 820, height: 360, margin: { l: 66, r: 24, t: 14, b: 54 },
        x: { domain: [0, 1], label: "diffusion step t / T", ticks: [0, 0.25, 0.5, 0.75, 1] }, y: { domain: [0, 1.02], label: "ᾱ (signal left)", ticks: [0, 0.25, 0.5, 0.75, 1] },
        series: [{ name: "linear", color: COL.amber, points: cl, width: 4 }, { name: "cosine", color: COL.cyan, points: cc, width: 4 }],
        vbands: [{ x0: 0.77, x1: 1, color: "var(--bad)", opacity: 0.08 }],
        vlines: [{ x: u, color: COL.text, width: 2 }],
        markers: [{ x: u, y: al, color: COL.amber, r: 9 }, { x: u, y: ac, color: COL.cyan, r: 9 }], legend: true,
        texts: [{ x: 0.78, y: 0.93, text: "pure noise (linear)", color: COL.coral, anchor: "start" }],
      });
    });
    $("n-slider").addEventListener("input", (e) => { t = +e.target.value; render(); });
    $("n-slider").addEventListener("change", (e) => e.target.blur());
    const play = () => {
      if (playId) { cancelAnimationFrame(playId); playId = null; return; }
      const t0 = performance.now(), start = t >= T ? 0 : t;
      const step = (now) => { t = Math.min(T, Math.round(start + (now - t0) / 5000 * T)); render(); playId = t < T ? requestAnimationFrame(step) : null; };
      playId = requestAnimationFrame(step);
    };
    const preset = (M.presets || [])[0];
    if (preset) { const im = new Image(); im.onload = () => { x0 = D.imageToArray(im, SIZE); drawFilms(); render(); }; im.src = preset.src; }
    drawFilms();
    hooks["s-noise"] = {
      enter: render,
      up: () => { t = Math.min(T, t + 50); render(); }, down: () => { t = Math.max(0, t - 50); render(); },
      play, reset: () => { t = 350; render(); },
    };
    render();
  })();

  // ============================================================ 2 · GENERATE
  (function sceneGen() {
    const st = { K: 50, var: "learned", seed: 1, es: null, anim: null, src: "recorded" };
    const blank = R["cifar_K50_learned"] && R["cifar_K50_learned"][0];
    const show = (fr, total) => {
      if (fr.x) $("g-x").src = fr.x;
      const empty = $("g-x0-empty"); if (empty) empty.style.display = fr.x0 ? "none" : "grid";
      if (fr.x0) { $("g-x0").src = fr.x0; } else if (fr.i < 0) { $("g-x0").src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"; }
      $("g-t").textContent = `t = ${fr.t}`;
      $("g-step").textContent = `step ${Math.max(0, fr.i + 1)} of ${fr.K}`;
      $("g-el").textContent = `${(fr.elapsed || 0).toFixed(1)} s`;
      $("g-prog").style.width = (100 * Math.max(0, fr.i + 1) / fr.K).toFixed(1) + "%";
    };
    const stop = () => { if (st.es) { st.es.close(); st.es = null; } if (st.anim) { clearTimeout(st.anim); st.anim = null; } };
    const msg = (html) => { $("g-msg").innerHTML = html; };
    const describe = () => `Trained with T = 4000 steps, sampled with K = ${st.K} (${Math.round(4000 / st.K)}× fewer network calls), using ${st.var === "learned" ? "the learned variance Σθ" : "the fixed variance σ² = β"}.`;
    function replay(note) {
      stop();
      const frames = R[`cifar_K${st.K}_${st.var}`];
      st.src = "recorded";
      $("g-src").textContent = "recorded run"; $("g-src").className = "note";
      if (!frames) { msg(`No recording for K = ${st.K} with this variance. Start the GPU server for live generation.`); return; }
      msg((note ? note + "<br>" : "") + describe());
      const total = Math.min(5, Math.max(1.6, frames[frames.length - 1].elapsed));
      let k = 0;
      const tick = () => { show(frames[k]); k++; if (k < frames.length) st.anim = setTimeout(tick, (total * 1000) / frames.length); else done(frames[frames.length - 1]); };
      tick();
    }
    function done(last) {
      msg(describe() + (st.var === "fixed_large" && st.K <= 50 ? `<br><span class="c">The images keep visible noise:</span> σ² = β adds too much noise on the last steps.` : (st.src === "live" ? `<br>Finished in <b>${(last.elapsed || 0).toFixed(1)}&nbsp;s</b> for a batch of 4 images.` : `<br>Recorded on this laptop: ${(last.elapsed || 0).toFixed(1)}&nbsp;s for a batch of 4 images.`)));
    }
    function liveGen() {
      if (!live.ok) { replay(`Offline: playing the run recorded on this laptop (same code, seed 0).`); return; }
      stop();
      st.seed++;
      st.src = "live";
      $("g-src").textContent = "running on the GPU"; $("g-src").className = "note";
      msg(`Generating on the GPU. ${describe()}`);
      let last = null, stall = null;
      const arm = () => { clearTimeout(stall); stall = setTimeout(() => { stop(); replay(`The live stream stalled, so this is the recorded run.`); }, 9000); };
      arm();
      const es = new EventSource(`api/generate?model=cifar&K=${st.K}&var=${st.var}&seed=${st.seed}&n=4`);
      st.es = es;
      es.onmessage = (ev) => { last = JSON.parse(ev.data); show(last); arm(); };
      es.addEventListener("done", () => { clearTimeout(stall); stop(); if (last) done(last); });
      es.onerror = () => { clearTimeout(stall); if (st.es) { stop(); if (!last || last.i < st.K - 1) replay(`The live server returned an error, so this is the recorded run.`); } };
    }
    segc("g-k", (v) => { st.K = +v; stop(); reset(); });
    segc("g-var", (v) => { st.var = v; stop(); reset(); });
    $("g-live").onclick = liveGen;
    $("g-replay").onclick = () => replay();
    function reset() {
      const frames = R[`cifar_K${st.K}_${st.var}`] || R["cifar_K50_learned"];
      if (frames) show(frames[0]);
      $("g-prog").style.width = "0%";
      msg(describe());
    }
    hooks["s-gen"] = {
      enter: () => { if (!st.es && !st.anim) reset(); },
      up: () => { const ks = [10, 50, 250], i = Math.min(ks.length - 1, ks.indexOf(st.K) + 1); st.K = ks[i]; setSeg("g-k", st.K); stop(); reset(); },
      down: () => { const ks = [10, 50, 250], i = Math.max(0, ks.indexOf(st.K) - 1); st.K = ks[i]; setSeg("g-k", st.K); stop(); reset(); },
      v: () => { st.var = st.var === "learned" ? "fixed_large" : "learned"; setSeg("g-var", st.var); stop(); reset(); },
      go: liveGen, reset: () => { stop(); reset(); },
    };
    if (blank) show(blank);
  })();

  // ============================================================ 3 · SPEED
  (function sceneSpeed() {
    const S = M.samples || {};
    const Ks = Object.keys(S).map(Number).filter((k) => S[k].learned).sort((a, b) => a - b);
    const modes = [
      { v: "fixed_large", name: "Fixed σ² = β", short: "fixed σ² = β", tag: "DDPM default", color: "var(--base)" },
      { v: "fixed_small", name: `Fixed σ² = ${BT}`, short: `fixed σ² = ${BT}`, tag: "posterior variance", color: "var(--alt2)" },
      { v: "ddim", name: "DDIM (η = 0)", short: "DDIM", tag: "Song et al., 2020", color: "var(--alt)" },
    ];
    const st = { K: Ks.includes(10) ? 10 : Ks[0], mode: 0 };
    $("sp-k").innerHTML = Ks.map((k) => `<button data-v="${k}">${k}</button>`).join("");
    const F = M.fid, L = M.latency || {};
    const render = () => {
      setSeg("sp-k", st.K);
      const md = modes[st.mode];
      const A = $("sp-a"), B = $("sp-b");
      [A, B].forEach((im) => (im.style.opacity = 0.2));
      setTimeout(() => {
        const bsrc = (S[st.K] || {})[md.v];
        A.src = (S[st.K] || {}).learned || "";
        B.src = bsrc || "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
        B.style.background = bsrc ? "#000" : "var(--bg-2)";
        $("sp-kb").textContent = `K = ${st.K}`;
        $("sp-b-empty").style.display = bsrc ? "none" : "grid";
        [A, B].forEach((im) => (im.style.opacity = 1));
      }, 90);
      $("sp-ka").textContent = `K = ${st.K}`; $("sp-kb").textContent = `K = ${st.K}`;
      $("sp-bname").innerHTML = md.name; $("sp-bname").style.color = md.color;
      $("sp-btag").textContent = md.tag; $("sp-btag").style.color = md.color; $("sp-btag").style.borderColor = md.color;
      // FID chart
      if (F && F.curves && F.curves.learned) {
        $("sp-fidtag").textContent = `FID, ${F.n} samples per point`;
        const colors = { learned: COL.cyan, fixed_large: COL.amber, fixed_small: COL.blue, ddim: COL.violet };
        const names = { learned: "learned Σθ", fixed_large: "fixed σ² = β", fixed_small: `fixed σ² = ${BT}`, ddim: "DDIM" };
        const series = Object.keys(F.curves).map((s) => {
          const hot = s === "learned" || s === md.v;
          return { name: names[s], color: colors[s], points: F.curves[s], width: hot ? 4.5 : 2, opacity: hot ? 1 : 0.35, dots: hot, r: 6 };
        });
        const ys = series.flatMap((s) => s.points.map((p) => p[1]));
        const lo = Math.min(...ys, F.floor || 1e9) * 0.85, hi = Math.max(...ys) * 1.2;
        if (F.floor) series.push({ name: "real test images (same n)", color: "#444", points: [[8, F.floor], [320, F.floor]], dash: "8 6", width: 2.5 });
        D.chart($("sp-fid"), {
          width: 840, height: 300, margin: { l: 78, r: 20, t: 10, b: 58 },
          x: { domain: [8, 320], log: true, label: "sampling steps K", ticks: [10, 25, 50, 100, 250], format: (v) => String(v) },
          y: { domain: [lo, hi], log: true, label: "FID (lower is better)", ticks: [30, 40, 60, 100, 150, 250].filter((t) => t >= lo && t <= hi), format: (v) => String(v) },
          series, vlines: st.K <= 300 ? [{ x: st.K, color: COL.text, width: 2 }] : [], legend: true,
        });
      } else {
        $("sp-fidtag").textContent = "";
        $("sp-fid").innerHTML = `<p style="font-size:22px;color:#555;margin:4px 0 8px">FID results pending. The samples on the left support a qualitative comparison.</p>`;
      }
      // latency chart
      const lk = [10, 25, 50, 100, 250, 1000, 4000].filter((k) => L["K" + k]);
      if (lk.length) {
        const one = L.one_forward_pass || L["K" + lk[0]] / lk[0];
        D.chart($("sp-lat"), {
          width: 840, height: F && F.curves ? 250 : 380, margin: { l: 78, r: 24, t: 14, b: 60 },
          x: { domain: [8, 5000], log: true, label: "sampling steps K (batch 1, laptop GPU)", ticks: [10, 50, 250, 1000, 4000], format: (v) => String(v) },
          y: { domain: [one * 0.6, 400], log: true, label: "s / image", ticks: [0.1, 1, 10, 100] },
          series: [{ name: "measured", color: COL.cyan, points: lk.map((k) => [k, L["K" + k]]), dots: true, r: 6, width: 4, labels: lk.map((k) => (k === 10 ? "" : (L["K" + k] < 10 ? L["K" + k].toFixed(1) : Math.round(L["K" + k])) + " s")), labelDx: -46, labelDy: -16 }],
          vlines: [{ x: st.K, color: COL.text, width: 2 }],
          texts: [{ x: 12, y: one * 1.25, text: `one network call: ${(one * 1000).toFixed(0)} ms (a GAN needs one call per image)`, color: COL.coral }],
        });
      }
      // banner
      const lat = L["K" + st.K], slow = L.K4000;
      const fidA = F && F.curves && F.curves.learned ? (F.curves.learned.find((p) => p[0] === st.K) || [])[1] : null;
      const fidB = F && F.curves && F.curves[md.v] ? (F.curves[md.v].find((p) => p[0] === st.K) || [])[1] : null;
      $("sp-banner").innerHTML = (st.K >= 4000 ? `<b>K = 4000</b> is the full training schedule.` : `With <b>K = ${st.K}</b> the model makes ${Math.round(4000 / st.K)}× fewer network calls than with all 4000 steps.`) +
        (lat ? ` Measured: <b>${lat < 10 ? lat.toFixed(1) : Math.round(lat)}&nbsp;s</b> per image` + (slow && st.K < 4000 ? `, against ${Math.round(slow)}&nbsp;s at 4000 steps.` : ".") : "") +
        (fidA ? `<br>FID at this K: learned Σθ <b style="color:var(--ours)">${fidA.toFixed(1)}</b>, ${md.short} <b style="color:${md.color}">${fidB ? fidB.toFixed(1) : "not run"}</b>.` : "");
    };
    segc("sp-k", (v) => { st.K = +v; render(); });
    hooks["s-speed"] = {
      enter: render,
      up: () => { const i = Math.min(Ks.length - 1, Ks.indexOf(st.K) + 1); st.K = Ks[i]; render(); },
      down: () => { const i = Math.max(0, Ks.indexOf(st.K) - 1); st.K = Ks[i]; render(); },
      v: () => { st.mode = (st.mode + 1) % modes.length; render(); },
      reset: () => { st.K = Ks.includes(10) ? 10 : Ks[0]; st.mode = 0; render(); },
    };
  })();

  // ============================================================ 4 · SHIFT
  let oodAuc = null;
  (function sceneOOD() {
    const O = M.ood;
    if (!O || !O.values || !O.values.cifar_hybrid) return;
    const v = O.values.cifar_hybrid;
    const c = v.cifar10_test, s = v.svhn_test;
    oodAuc = O.auroc;
    const lo = Math.floor(Math.min(...c, ...s) * 10) / 10, hi = Math.ceil(Math.max(...c, ...s) * 10) / 10;
    const nb = 34, w = (hi - lo) / nb;
    const hist = (a) => { const h = new Array(nb).fill(0); a.forEach((x) => { h[Math.min(nb - 1, Math.max(0, Math.floor((x - lo) / w)))]++; }); return h.map((n, i) => [lo + (i + 0.5) * w, 100 * n / a.length]); };
    const hc = hist(c), hs = hist(s);
    const ymax = Math.max(...hc.map((p) => p[1]), ...hs.map((p) => p[1])) * 1.15;
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const box = $("o-hists");
    box.innerHTML = `
      <div class="ex-row"><div><div style="font:700 24px var(--display);color:var(--c-in)">CIFAR-10 test</div><div style="color:var(--faint);font-size:18px">in-distribution · n=${c.length}</div></div><img src="${(O.examples || {}).cifar10_test || ""}" alt="CIFAR-10 examples"></div>
      <div id="o-h1"></div>
      <div class="ex-row" style="margin-top:6px"><div><div style="font:700 24px var(--display);color:var(--c-out)">SVHN test</div><div style="color:var(--faint);font-size:18px">never seen · n=${s.length}</div></div><img src="${(O.examples || {}).svhn_test || ""}" alt="SVHN examples"></div>
      <div id="o-h2"></div>`;
    const common = { width: 980, height: 200, margin: { l: 70, r: 20, t: 8, b: 40 }, x: { domain: [lo, hi], label: "" }, y: { domain: [0, ymax], label: "% of images", nticks: 3 } };
    D.chart($("o-h1"), Object.assign({}, common, { series: [{ color: "var(--c-in)", points: hc, bars: true, barWidth: 980 / nb * 0.82 }], vlines: [{ x: mean(c), color: "var(--c-in)", label: `mean ${mean(c).toFixed(2)}` }] }));
    D.chart($("o-h2"), Object.assign({}, common, { x: { domain: [lo, hi], label: "estimated bits/dim (variational NLL bound); lower means the model rates the image as more likely" }, height: 240, margin: { l: 70, r: 20, t: 8, b: 62 },
      series: [{ color: "var(--c-out)", points: hs, bars: true, barWidth: 980 / nb * 0.82 }], vlines: [{ x: mean(s), color: "var(--c-out)", label: `mean ${mean(s).toFixed(2)}` }] }));
    $("o-auc").textContent = O.auroc.toFixed(2);
    $("o-auc").style.color = O.auroc < 0.6 ? "var(--bad)" : O.auroc < 0.8 ? "var(--base)" : "var(--good)";
    const aa = O.auroc_all && O.auroc_all.cifar_vlb && O.auroc_all.cifar_vlb.svhn_test;
    $("o-auc2").innerHTML = aa != null ? `L<sub>vlb</sub> checkpoint (best NLL): ${aa.toFixed(2)}` : "";
    // probe strip: simple synthetic images vs real photos
    const probes = (O.probes || []).slice().sort((a, b) => a.bpd - b.bpd);
    const renderProbes = (extra) => {
      const keep = ["plain black", "smooth gradient", "SVHN digit", "CIFAR-10 photo", "checker board", "random pixels"];
      const all = probes.filter((p) => keep.includes(p.name)).concat(extra || []).sort((a, b) => a.bpd - b.bpd).slice(0, 7);
      $("o-probes").innerHTML = all.map((p) => `<div class="probe ${p.mine ? "mine" : /CIFAR-10 photo/.test(p.name) ? "real" : "odd"}"><img src="${p.img}" alt="${p.name}"><b>${p.bpd < 10 ? p.bpd.toFixed(2) : p.bpd.toFixed(1)}</b><span>${p.name}</span></div>`).join("");
    };
    renderProbes();
    const mine = [];
    $("o-up").addEventListener("change", (e) => {
      const f = e.target.files[0];
      if (!f) return;
      if (!live.ok) { $("o-upmsg").textContent = "start the GPU server (Run_Live_Demo.bat) to score new images"; return; }
      const rd = new FileReader();
      rd.onload = () => {
        $("o-upmsg").textContent = "scoring on the GPU (about 5 s)";
        fetch("api/bpd", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: rd.result }) })
          .then((r) => r.json()).then((j) => {
            if (j.bpd == null) throw new Error(j.detail || "error");
            mine.push({ name: "your image", bpd: j.bpd, img: j.thumb, mine: true });
            renderProbes(mine);
            $("o-upmsg").innerHTML = `your image: <b style="color:var(--good)">${j.bpd.toFixed(2)} bits/dim</b> (${j.seconds}s)`;
          }).catch((err) => { $("o-upmsg").textContent = "scoring failed: " + err.message; });
      };
      rd.readAsDataURL(f);
      e.target.value = "";
    });
    // shift table: mean bits/dim and AUROC of "high bits/dim = unfamiliar" against CIFAR-10 test
    const rowsT = [
      ["cifar10_test", "CIFAR-10 test (reference)", null],
      ["cifar10_train", "CIFAR-10 train (seen in training)", "near chance, as it should be"],
      ["cifar10_gaussnoise", "CIFAR-10 + heavy pixel noise", "caught"],
      ["cifar10_vflip", "CIFAR-10 upside down", "chance"],
      ["svhn_test", "SVHN house numbers", "backwards"],
    ].filter((r) => v[r[0]]);
    const aa2 = (O.auroc_all && O.auroc_all.cifar_hybrid) || {};
    $("o-bars").innerHTML = `<table class="shift"><tr><th>Input</th><th class="n">bits/dim</th><th class="n">AUROC</th><th></th></tr>` +
      rowsT.map(([k, label, note]) => {
        const a = aa2[k];
        const col = a == null ? "" : a > 0.9 ? "var(--good)" : a < 0.35 ? "var(--bad)" : "var(--muted)";
        return `<tr${k === "svhn_test" ? ' class="hl"' : ""}><td>${label}</td><td class="n">${mean(v[k]).toFixed(2)}</td><td class="n" style="color:${col}">${a == null ? "" : a.toFixed(2)}</td><td class="note2">${note || ""}</td></tr>`;
      }).join("") + `</table>`;
    const lower = mean(s) < mean(c);
    $("o-banner").innerHTML = lower
      ? `Likelihood catches pixel-level corruption but not a change in content: SVHN digits look <span class="c">more</span> likely than real CIFAR-10 images. The paper’s best-likelihood model (L<sub>vlb</sub>) is also backwards on SVHN.`
      : `Here SVHN gets higher bits/dim (AUROC ${O.auroc.toFixed(2)}), so likelihood separates this pair; the other shifts on the left behave differently.`;
  })();

  // ============================================================ 5 · VERDICT
  (function sceneVerdict() {
    const auc = oodAuc != null ? oodAuc.toFixed(2) : "?";
    const lat50 = M.latency && M.latency.K50 ? M.latency.K50.toFixed(1) : "2.1";
    const one = M.latency && M.latency.one_forward_pass ? Math.round(M.latency.one_forward_pass * 1000) : 44;
    const rows = [
      ["50 to 100 sampling steps with little loss in quality", "Fig. 8, and our run: at K = 50, FID 34 with learned Σθ against 87 with fixed σ² = β (real images: 30 at this sample size).", ["ok", "Supported"], `Each image still costs about 50 network calls: ${lat50} s per image here, against ${one} ms for one call.`],
      ["Learned Σθ and the hybrid loss give a competitive likelihood", "Tables 1 to 3. The best NLL (L<sub>vlb</sub>) and the best FID (L<sub>hybrid</sub>) come from <b>different models</b>.", ["part", "Partly"], "Report NLL and FID for the model you actually ship."],
      ["Higher recall than GANs, so better coverage", "Table 4: recall 0.71 versus 0.59 for one BigGAN-deep baseline; one run.", ["part", "Partly"], "Coverage of classes is not coverage of people or groups; memorized copies also count as recall."],
      ["Implied: a good likelihood can flag unfamiliar inputs", `Our check: trained on CIFAR-10, scored on SVHN, AUROC ${auc}.`, [oodAuc != null && oodAuc < 0.6 ? "no" : "part", oodAuc != null && oodAuc < 0.6 ? "Fails the CIFAR-10 to SVHN test" : "Mixed"], "Do not use NLL as an anomaly or data‑validation gate."],
    ];
    $("v-table").innerHTML = `<tr><th>Claim</th><th>Evidence</th><th>Verdict</th><th>For deployment</th></tr>` +
      rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td><span class="chip ${r[2][0]}">${r[2][1]}</span></td><td>${r[3]}</td></tr>`).join("");
  })();

  // ============================================================ backups
  (function sceneMem() {
    const N = M.nn;
    if (!N) return;
    $("m-pairs").innerHTML = `<h3>Closest 6 samples (left) and their nearest training image (right)</h3>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:22px">${N.pairs.map((p) => `<div><div class="nnpair"><img src="${p.gen}" alt="generated"><img src="${p.nn}" alt="nearest training image"></div><div style="font:600 19px var(--mono);color:var(--muted);margin-top:6px;text-align:center">L2 = ${p.d.toFixed(2)}</div></div>`).join("")}</div>`;
    const g = N.gen_dist, r = N.test_ref_dist;
    const lo = 0, hi = Math.ceil(Math.max(...g, ...r));
    const nb = 24, w = (hi - lo) / nb;
    const hist = (a) => { const h = new Array(nb).fill(0); a.forEach((x) => h[Math.min(nb - 1, Math.floor((x - lo) / w))]++); return h.map((n, i) => [lo + (i + 0.5) * w, 100 * n / a.length]); };
    const hg = hist(g), hr = hist(r);
    D.chart($("m-hist"), { width: 700, height: 360, margin: { l: 66, r: 16, t: 10, b: 54 }, x: { domain: [lo, hi], label: "L2 distance to nearest training image (pixels in [0,1])" }, y: { domain: [0, Math.max(...hg.map((p) => p[1]), ...hr.map((p) => p[1])) * 1.15], label: "% of images" },
      series: [{ name: "generated samples", color: COL.cyan, points: hg.map((p) => [p[0] - w * 0.25, p[1]]), bars: true, barWidth: 10, opacity: 0.9 }, { name: "held-out test images (reference)", color: "#555", points: hr.map((p) => [p[0] + w * 0.25, p[1]]), bars: true, barWidth: 10, opacity: 0.85 }], legend: true });
    const med = (a) => { const b = a.slice().sort((x, y) => x - y), n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; };
    $("m-banner").innerHTML = `Median distance: generated samples <b>${med(g).toFixed(2)}</b>, held-out real test images ${med(r).toFixed(2)}. ${med(g) < med(r) ? "The samples sit slightly closer to training images than unseen photos do," : "No sign of direct copying,"} but even the six closest pairs are different images. A check on ${g.length} samples cannot rule out memorization; Carlini et al. used one million.`;
  })();

  (function sceneWhy() {
    const T = 4000, abar = D.cumprod(D.cosineBetas(T));
    const draw = (K) => {
      setSeg("w-k", K);
      const steps = D.spaceTimesteps(T, K), b = D.respace(abar, steps), pv = D.posteriorVariance(b);
      const idx = D.downsampleIdx(steps.length - 1, 300).map((j) => j + 1);
      D.chart($("w-chart"), {
        width: 1000, height: 560, margin: { l: 80, r: 24, t: 14, b: 60 },
        x: { domain: [1, 4000], log: true, label: "timestep t (log scale)", ticks: [1, 10, 100, 1000, 4000], format: (v) => String(v) },
        y: { domain: [1e-6, 1.2], log: true, label: "variance of one reverse step" },
        series: [{ name: "β (upper bound)", color: COL.amber, points: idx.map((j) => [Math.max(1, steps[j]), b[j]]), width: 4, dots: K <= 50, r: 5 },
                 { name: `${BT} (posterior variance, lower bound)`, color: "var(--alt2)", points: idx.map((j) => [Math.max(1, steps[j]), pv[j]]), width: 4, dots: K <= 50, r: 5 }], legend: true,
      });
      const ratio = b[1] / pv[1]; $("w-ratio").textContent = (ratio >= 100 ? Math.round(ratio).toLocaleString("en-US") : ratio.toFixed(1)) + "×";
    };
    segc("w-k", (v) => draw(+v));
    hooks["s-why"] = { enter: () => {} };
    draw(50);
  })();

  (function sceneImagenet() {
    const I = M.imagenet || {};
    const cells = [
      [25, "learned", "25 steps, learned Σθ", "var(--ours)"], [25, "fixed_large", "25 steps, fixed σ² = β", "var(--base)"], [25, "ddim", "25 steps, DDIM", "var(--alt)"],
      [50, "learned", "50 steps, learned Σθ", "var(--ours)"], [100, "learned", "100 steps, learned Σθ", "var(--ours)"], [250, "learned", "250 steps, learned Σθ", "var(--ours)"],
    ].filter(([k, s2]) => I[k] && I[k][s2]);
    if (!cells.length) return;
    $("i-grid").innerHTML = cells.map(([k, s2, lab, col]) => `<div class="panel"><h3 style="font-size:22px;color:${col};margin-bottom:8px">${lab}</h3><img class="px" style="aspect-ratio:auto" src="${I[k][s2]}" alt="${lab}"></div>`).join("") +
      `<p class="banner" style="grid-column:1/-1;font-size:22px;margin:0">At 25 steps the fixed variance leaves noise and the learned variance does not. DDIM is sharper still at 25 steps, which matches the paper's own finding that DDIM wins below about 50 steps (Fig. 8).</p>`;
  })();

  // ------------------------------------------------------------ keyboard
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
    const h = hooks[scenes[cur].id] || {};
    const k = e.key;
    if (k === "ArrowRight" || k === "PageDown" || (k === " " && e.target === document.body)) { e.preventDefault(); go(cur + 1); }
    else if (k === "ArrowLeft" || k === "PageUp") { e.preventDefault(); go(cur - 1); }
    else if (k === "ArrowUp") { e.preventDefault(); h.up && h.up(); }
    else if (k === "ArrowDown") { e.preventDefault(); h.down && h.down(); }
    else if (/^[1-9]$/.test(k)) { go(+k - 1); }
    else if (k === "b" || k === "B") { go(mainCount); }
    else if (k === "f" || k === "F") { if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {}); else document.exitFullscreen(); }
    else if (k === "t" || k === "T") toggleTimer();
    else if (k === "g" || k === "G") { if (scenes[cur].id === "s-gen" && h.go) h.go(); }
    else if (k === "p" || k === "P") { h.play && h.play(); }
    else if (k === "v" || k === "V") { h.v && h.v(); }
    else if (k === "r" || k === "R") { h.reset && h.reset(); }
  });

  const start = scenes.findIndex((s) => "#" + s.id === location.hash);
  go(start >= 0 ? start : 0);
})();
