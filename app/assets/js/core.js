/* core.js — shared math + tiny SVG chart library for the Improved DDPM visualizer.
   No dependencies. Everything mirrors openai/improved-diffusion (gaussian_diffusion.py, respace.py). */
(function (global) {
  "use strict";

  // ---------- seeded RNG ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussians(n, seed) {
    const rnd = mulberry32(seed);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      let u = 0, v = 0;
      while (u === 0) u = rnd();
      v = rnd();
      const r = Math.sqrt(-2 * Math.log(u));
      out[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return out;
  }

  // ---------- noise schedules (exactly as in the released code) ----------
  function linearBetas(T) {
    const scale = 1000 / T, b0 = scale * 0.0001, b1 = scale * 0.02;
    const out = new Float64Array(T);
    for (let i = 0; i < T; i++) out[i] = T === 1 ? b0 : b0 + (b1 - b0) * i / (T - 1);
    return out;
  }
  function cosineAlphaBar(u, s) {
    s = s === undefined ? 0.008 : s;
    const c = Math.cos(((u + s) / (1 + s)) * Math.PI / 2);
    return c * c;
  }
  function cosineBetas(T, s) {
    const out = new Float64Array(T);
    for (let i = 0; i < T; i++) {
      const t1 = i / T, t2 = (i + 1) / T;
      out[i] = Math.min(1 - cosineAlphaBar(t2, s) / cosineAlphaBar(t1, s), 0.999);
    }
    return out;
  }
  function betasFor(name, T, s) { return name === "cosine" ? cosineBetas(T, s) : linearBetas(T); }
  function cumprod(betas) {
    const out = new Float64Array(betas.length);
    let p = 1;
    for (let i = 0; i < betas.length; i++) { p *= 1 - betas[i]; out[i] = p; }
    return out;
  }
  // posterior variance  beta~_t = beta_t (1 - abar_{t-1}) / (1 - abar_t)   (Eq. 10)
  function posteriorVariance(betas) {
    const ab = cumprod(betas);
    const out = new Float64Array(betas.length);
    for (let i = 0; i < betas.length; i++) {
      const prev = i === 0 ? 1 : ab[i - 1];
      out[i] = betas[i] * (1 - prev) / (1 - ab[i]);
    }
    return out;
  }
  // respace.py: space_timesteps(T, [K]) — K evenly spaced (rounded) indices in [0, T-1]
  function spaceTimesteps(T, K) {
    if (K >= T) return Array.from({ length: T }, (_, i) => i);
    const frac = K <= 1 ? 1 : (T - 1) / (K - 1);
    const out = [];
    let cur = 0;
    const roundEven = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };
    for (let i = 0; i < K; i++) { out.push(roundEven(cur)); cur += frac; }
    return Array.from(new Set(out)).sort((a, b) => a - b);
  }
  // DDIM-style constant stride: range(0, T, stride) with exactly K elements
  function ddimTimesteps(T, K) {
    for (let i = 1; i < T; i++) {
      const n = Math.ceil(T / i);
      if (n === K) return Array.from({ length: n }, (_, j) => j * i);
    }
    // fall back to nearest stride
    const stride = Math.max(1, Math.round(T / K));
    const out = [];
    for (let t = 0; t < T; t += stride) out.push(t);
    return out;
  }
  // Respaced process (Eq. 19): beta_{S_t} = 1 - abar_{S_t}/abar_{S_{t-1}}
  function respace(alphaBar, steps) {
    const betas = new Float64Array(steps.length);
    let last = 1;
    steps.forEach((t, i) => { betas[i] = 1 - alphaBar[t] / last; last = alphaBar[t]; });
    return betas;
  }

  // ---------- small helpers ----------
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  function fmt(x, d) {
    d = d === undefined ? 2 : d;
    if (!isFinite(x)) return "–";
    const ax = Math.abs(x);
    if (ax !== 0 && (ax < 1e-3 || ax >= 1e5)) {
      const e = Math.floor(Math.log10(ax));
      return (x / Math.pow(10, e)).toFixed(1) + "×10" + supNum(e);
    }
    return x.toFixed(d);
  }
  function supNum(n) {
    const map = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
    return String(n).split("").map(c => map[c] || c).join("");
  }
  function downsampleIdx(n, max) {
    if (n <= max) return Array.from({ length: n }, (_, i) => i);
    const out = [];
    for (let i = 0; i < max; i++) out.push(Math.round(i * (n - 1) / (max - 1)));
    return out;
  }

  // ---------- image helpers ----------
  // Load an <img>/canvas into a Float32Array in [-1,1], shape [H*W*3]
  function imageToArray(src, size) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const g = c.getContext("2d");
    g.imageSmoothingQuality = "high";
    // center-crop to square
    const w = src.naturalWidth || src.width, h = src.naturalHeight || src.height;
    const m = Math.min(w, h);
    g.drawImage(src, (w - m) / 2, (h - m) / 2, m, m, 0, 0, size, size);
    const d = g.getImageData(0, 0, size, size).data;
    const out = new Float32Array(size * size * 3);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
      out[j] = d[i] / 127.5 - 1; out[j + 1] = d[i + 1] / 127.5 - 1; out[j + 2] = d[i + 2] / 127.5 - 1;
    }
    return out;
  }
  // x_t = sqrt(abar) x0 + sqrt(1-abar) eps   (Eq. 9) — drawn into a canvas
  function drawNoised(canvas, x0, eps, abar, size) {
    canvas.width = canvas.height = size;
    const g = canvas.getContext("2d");
    const img = g.createImageData(size, size);
    const a = Math.sqrt(abar), b = Math.sqrt(1 - abar);
    for (let p = 0, j = 0; p < size * size; p++, j += 3) {
      const o = p * 4;
      img.data[o] = clamp((a * x0[j] + b * eps[j] + 1) * 127.5, 0, 255);
      img.data[o + 1] = clamp((a * x0[j + 1] + b * eps[j + 1] + 1) * 127.5, 0, 255);
      img.data[o + 2] = clamp((a * x0[j + 2] + b * eps[j + 2] + 1) * 127.5, 0, 255);
      img.data[o + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }
  // A procedural fallback picture (sunset over hills) so the page works with zero data files.
  function proceduralImage(size) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const g = c.getContext("2d");
    const sky = g.createLinearGradient(0, 0, 0, size);
    sky.addColorStop(0, "#1e3a8a"); sky.addColorStop(0.55, "#f97316"); sky.addColorStop(1, "#fde68a");
    g.fillStyle = sky; g.fillRect(0, 0, size, size);
    g.fillStyle = "#fff7d6"; g.beginPath(); g.arc(size * 0.68, size * 0.5, size * 0.13, 0, 7); g.fill();
    g.fillStyle = "#14532d"; g.beginPath(); g.moveTo(0, size * 0.72);
    for (let x = 0; x <= size; x++) g.lineTo(x, size * (0.7 - 0.08 * Math.sin(x / size * 6.5)));
    g.lineTo(size, size); g.lineTo(0, size); g.fill();
    g.fillStyle = "#052e16"; g.beginPath(); g.moveTo(0, size * 0.85);
    for (let x = 0; x <= size; x++) g.lineTo(x, size * (0.84 - 0.05 * Math.cos(x / size * 9)));
    g.lineTo(size, size); g.lineTo(0, size); g.fill();
    return c;
  }

  // ---------- tiny SVG chart ----------
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) {
      const v = attrs[k];
      if (v === undefined || v === null) continue;
      // CSS variables are not reliably honoured in SVG presentation attributes → route them through style
      if ((k === "fill" || k === "stroke") && typeof v === "string" && v.indexOf("var(") === 0) e.style.setProperty(k, v);
      else e.setAttribute(k, v);
    }
    if (parent) parent.appendChild(e);
    return e;
  }

  // Write text into an SVG <text>, turning Unicode superscripts (10⁻⁶) into a raised <tspan> with plain digits,
  // because the embedded fonts do not contain the superscript glyphs.
  const SUP = { "⁻": "\u2212", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9" };
  function setText(t, str) {
    str = String(str);
    if (!/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]/.test(str)) { t.textContent = str; return; }
    t.textContent = "";
    str.split(/([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/).forEach((part, i) => {
      if (!part) return;
      if (i % 2 === 1) {
        const up = el("tspan", { dy: "-0.45em", "font-size": "72%" }, t); up.textContent = part.split("").map((c) => SUP[c]).join("");
        const back = el("tspan", { dy: "0.45em" }, t); back.textContent = "\u200b";
      } else {
        const n = el("tspan", {}, t); n.textContent = part;
      }
    });
  }
  function makeScale(domain, range, log) {
    const [d0, d1] = domain, [r0, r1] = range;
    if (log) {
      const l0 = Math.log10(d0), l1 = Math.log10(d1);
      const f = v => r0 + (Math.log10(Math.max(v, 1e-300)) - l0) / (l1 - l0) * (r1 - r0);
      f.inv = p => Math.pow(10, l0 + (p - r0) / (r1 - r0) * (l1 - l0));
      return f;
    }
    const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
    f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
    return f;
  }
  function niceTicks(d0, d1, n) {
    const span = d1 - d0, step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(d0 / step) * step; v <= d1 + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function logTicks(d0, d1) {
    const out = [];
    for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
      const v = Math.pow(10, e);
      if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
    }
    return out;
  }
  function tickLabel(v, log) {
    if (log) {
      const e = Math.round(Math.log10(v));
      if (e >= -2 && e <= 3) return String(+v.toPrecision(3));
      return "10" + supNum(e);
    }
    if (Math.abs(v) >= 1e4) return v.toExponential(0).replace("e+", "e");
    return String(+v.toPrecision(4));
  }

  /* spec = {
       width, height, margin:{l,r,t,b},
       x:{domain:[a,b], log, label, ticks:[...], format(v)}, y:{...},
       series:[{name, color, points:[[x,y],...], dash, width, area, dots, labels}],
       vlines:[{x, color, label}], hbands:[{y0,y1,color}], legend:true|'bottom'
     } */
  function chart(container, spec) {
    container.innerHTML = "";
    const W = spec.width || 640, H = spec.height || 320;
    const m = Object.assign({ l: 58, r: 18, t: 16, b: 46 }, spec.margin || {});
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": spec.aria || "" }, container);
    const xs = makeScale(spec.x.domain, [m.l, W - m.r], spec.x.log);
    const ys = makeScale(spec.y.domain, [H - m.b, m.t], spec.y.log);
    const gGrid = el("g", { class: "grid" }, svg);
    const xt = spec.x.ticks || (spec.x.log ? logTicks(...spec.x.domain) : niceTicks(...spec.x.domain, spec.x.nticks || 6));
    const yt = spec.y.ticks || (spec.y.log ? logTicks(...spec.y.domain) : niceTicks(...spec.y.domain, spec.y.nticks || 5));
    // bands are clamped to the plotting area so they never spill outside the axes
    const yLo = Math.min(...spec.y.domain), yHi = Math.max(...spec.y.domain), xLo = Math.min(...spec.x.domain), xHi = Math.max(...spec.x.domain);
    (spec.hbands || []).forEach(b => {
      const a = clamp(Math.min(b.y0, b.y1), yLo, yHi), c = clamp(Math.max(b.y0, b.y1), yLo, yHi);
      if (c > a) el("rect", { x: m.l, width: W - m.l - m.r, y: ys(c), height: Math.abs(ys(a) - ys(c)), fill: b.color, opacity: b.opacity || 0.12 }, gGrid);
    });
    (spec.vbands || []).forEach(b => {
      const a = clamp(Math.min(b.x0, b.x1), xLo, xHi), c = clamp(Math.max(b.x0, b.x1), xLo, xHi);
      if (c > a) el("rect", { y: m.t, height: H - m.t - m.b, x: xs(a), width: Math.abs(xs(c) - xs(a)), fill: b.color, opacity: b.opacity || 0.12 }, gGrid);
    });
    yt.forEach(v => {
      const y = ys(v);
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, gGrid);
      const t = el("text", { x: m.l - 8, y: y + 4, class: "tick", "text-anchor": "end" }, gGrid);
      setText(t, spec.y.format ? spec.y.format(v) : tickLabel(v, spec.y.log));
    });
    xt.forEach(v => {
      const x = xs(v);
      el("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, class: "gridline faint" }, gGrid);
      const t = el("text", { x, y: H - m.b + 20, class: "tick", "text-anchor": "middle" }, gGrid);
      setText(t, spec.x.format ? spec.x.format(v) : tickLabel(v, spec.x.log));
    });
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
    el("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axis" }, svg);
    if (spec.x.label) { const t = el("text", { x: (m.l + W - m.r) / 2, y: H - 6, class: "axlabel", "text-anchor": "middle" }, svg); t.textContent = spec.x.label; }
    if (spec.y.label) { const t = el("text", { x: 14, y: (m.t + H - m.b) / 2, class: "axlabel", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, svg); t.textContent = spec.y.label; }
    const clipId = "c" + Math.random().toString(36).slice(2);
    const cp = el("clipPath", { id: clipId }, el("defs", {}, svg));
    el("rect", { x: m.l, y: m.t - 2, width: W - m.l - m.r, height: H - m.t - m.b + 4 }, cp);
    const gData = el("g", { "clip-path": `url(#${clipId})` }, svg);
    (spec.series || []).forEach(s => {
      const pts = s.points.filter(p => isFinite(p[0]) && isFinite(p[1]) && (!spec.y.log || p[1] > 0) && (!spec.x.log || p[0] > 0));
      if (!pts.length) return;
      if (s.area) {
        const base = spec.y.log ? spec.y.domain[0] : (s.areaBase !== undefined ? s.areaBase : spec.y.domain[0]);
        const d = "M" + pts.map(p => `${xs(p[0]).toFixed(1)},${ys(p[1]).toFixed(1)}`).join("L") + `L${xs(pts[pts.length - 1][0]).toFixed(1)},${ys(base).toFixed(1)}L${xs(pts[0][0]).toFixed(1)},${ys(base).toFixed(1)}Z`;
        el("path", { d, fill: s.color, opacity: s.areaOpacity || 0.14, stroke: "none" }, gData);
      }
      if (s.bars) {
        const bw = s.barWidth || 6;
        pts.forEach(p => el("rect", { x: xs(p[0]) - bw / 2, width: bw, y: Math.min(ys(p[1]), ys(spec.y.domain[0])), height: Math.abs(ys(spec.y.domain[0]) - ys(p[1])), fill: s.color, opacity: s.opacity || 0.85, rx: 1.5 }, gData));
      } else if (!s.dotsOnly) {
        const d = "M" + pts.map(p => `${xs(p[0]).toFixed(1)},${ys(p[1]).toFixed(1)}`).join("L");
        el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.5, "stroke-dasharray": s.dash || null, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: s.opacity || 1 }, gData);
      }
      if (s.dots) pts.forEach((p, i) => {
        const c = el("circle", { cx: xs(p[0]), cy: ys(p[1]), r: s.r || 5, fill: s.hollow ? "var(--bg)" : s.color, stroke: s.color, "stroke-width": 2 }, svg);
        if (s.titles && s.titles[i]) { const ti = el("title", {}, c); ti.textContent = s.titles[i]; }
      });
      if (s.labels) pts.forEach((p, i) => {
        if (!s.labels[i]) return;
        const t = el("text", { x: xs(p[0]) + (s.labelDx || 8), y: ys(p[1]) + (s.labelDy || 4), class: "ptlabel", fill: s.labelColor || s.color }, svg);
        setText(t, s.labels[i]);
      });
    });
    (spec.vlines || []).forEach(v => {
      const x = xs(v.x);
      el("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: v.color || "var(--text)", "stroke-width": v.width || 1.5, "stroke-dasharray": v.dash || "4 4", opacity: 0.9 }, svg);
      if (v.label) { const t = el("text", { x: x + 6, y: m.t + 14, class: "ptlabel", fill: v.color || "var(--text)" }, svg); setText(t, v.label); }
    });
    (spec.markers || []).forEach(mk => {
      el("circle", { cx: xs(mk.x), cy: ys(mk.y), r: mk.r || 6, fill: mk.color, stroke: "var(--bg)", "stroke-width": 2 }, svg);
      if (mk.label) { const t = el("text", { x: xs(mk.x) + (mk.dx || 10), y: ys(mk.y) + (mk.dy || -10), class: "ptlabel strong", fill: mk.color }, svg); setText(t, mk.label); }
    });
    (spec.texts || []).forEach(tx => {
      const t = el("text", { x: xs(tx.x), y: ys(tx.y), class: "ptlabel " + (tx.cls || ""), fill: tx.color || "var(--muted)", "text-anchor": tx.anchor || "start" }, svg);
      setText(t, tx.text);
    });
    if (spec.legend) {
      const lg = document.createElement("div");
      lg.className = "legend";
      spec.series.filter(s => s.name).forEach(s => {
        const it = document.createElement("span");
        it.innerHTML = `<i style="background:${s.color};${s.dash ? "background:repeating-linear-gradient(90deg," + s.color + " 0 6px,transparent 6px 10px)" : ""}"></i>${s.name}`;
        lg.appendChild(it);
      });
      container.appendChild(lg);
    }
    return { svg, xs, ys, W, H, m };
  }

  global.DDPM = {
    mulberry32, gaussians, linearBetas, cosineBetas, cosineAlphaBar, betasFor, cumprod, posteriorVariance,
    spaceTimesteps, ddimTimesteps, respace, clamp, fmt, supNum, downsampleIdx,
    imageToArray, drawNoised, proceduralImage, chart, svgEl: el,
  };
})(window);
