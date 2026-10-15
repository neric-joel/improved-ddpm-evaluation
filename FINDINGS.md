# Findings: Improved DDPM (Nichol & Dhariwal, ICML 2021)

**Paper:** A. Q. Nichol and P. Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021, PMLR 139:8162-8171. arXiv:2102.09672 (v1, 18 Feb 2021). Code and checkpoints: github.com/openai/improved-diffusion (MIT license).

**How to read this file**
- Section numbers (Sec.), tables (Tab.), figures (Fig.) and appendices (App.) refer to the arXiv v1 text.
- Every external number was checked against the cited source on 2026-10-13. Anything I could not confirm is tagged **[UNVERIFIED]**.
- Back-of-envelope arithmetic that I did myself (not printed in any paper) is tagged **[my calc]**.

---

## 0. Summary

1. **The headline pairs two different models.** The abstract's promise of competitive likelihood with preserved sample quality combines numbers from two models. The likelihood numbers in Tab. 3 (2.94 bits/dim on CIFAR-10, 3.53 on ImageNet-64) come from the L_vlb models, whose FIDs are 11.47 and 40.1. The model with good samples (L_hybrid) gets 3.17 bits/dim on CIFAR-10, which is worse than PixelCNN (3.14) and Flow++ (3.08) in the same table.
2. **Train on one dataset, test on another, and the likelihood story breaks.** The public Improved-DDPM CIFAR-10 checkpoint gives *higher* likelihood to SVHN than to CIFAR-10 itself. Used as an out-of-distribution (OOD) detector, it scores AUROC 0.091, far below chance (0.5) (Heng et al., NeurIPS 2024). The paper's own Fig. 2 explains why. Most of the bound comes from the first few, almost noise-free diffusion steps, i.e. fine pixel detail. So bits/dim tracks how simple an image is, not what it shows.
3. **"Fast enough for deployment" is claimed but never measured.** The paper reports no latency, energy or throughput. Fifty steps is still 50 passes of a ~39-GFLOP UNet, about 2 TFLOPs per 64x64 image [my calc], against one pass for a GAN. DDIM beats the paper's method below 50 steps.
4. **The recall advantage over GANs rests on a thin baseline.** It is one BigGAN-deep the authors trained themselves for 125K iterations, run once. Recall is computed in an ImageNet-classifier feature space; the paper states that FID uses the training set as reference (Sec. 5), and if precision/recall used the same reference, a model that copies training images would also score well. Carlini et al. extracted 301 CIFAR-10 training images from 1M samples of the released Improved-DDPM checkpoint.
5. **The evaluation is thin throughout.** Every result is a single run with no error bars. The ImageNet ablations use 10K-sample FID, which the paper says is biased by about 2 FID points, larger than the ~1-point gaps it compares. The scaling "law" uses 4 model sizes with hand-picked asymptotes. The conclusion says more compute *trivially* helps, yet App. G reports overfitting even on ImageNet.

---

## A. Claim-by-claim evidence audit

### A.0 Summary table

| # | Headline claim (where) | Evidence the paper gives | Main weaknesses | Verdict |
|---|---|---|---|---|
| A1 | Competitive NLL *while maintaining* high sample quality (Abstract; Sec. 3; Tab. 1-3) | Tab. 3: 2.94 (CIFAR-10), 3.53 (ImageNet-64) bits/dim | Best NLL and best FID come from **different models** (L_vlb vs L_hybrid). About half the NLL gain comes from simply raising T from 1000 to 4000. NLL is dominated by imperceptible detail (Fig. 2, App. D) | Partially supported; the abstract overstates |
| A2 | Learned variances allow ~10x fewer steps with *negligible* quality loss (Abstract; Sec. 4; Fig. 8, 11, 15) | FID-vs-steps curves (25 to 4000 steps) | Shown only as FID curves, with no tables and no confidence intervals. DDIM wins below 50 steps. NLL does not get the speed-up (App. E). No wall-clock or energy numbers | Supported for FID at about 100 or more steps; deployment benefit not measured |
| A3 | Diffusion has much higher recall than GANs, i.e. better mode coverage (Intro; Sec. 5; Tab. 4; Conclusion) | Tab. 4: recall 0.71-0.72 vs 0.59 for BigGAN-deep | One self-trained BigGAN-deep (125K iterations, no truncation, single run). Inception-V3 features. Training set used as reference. The authors' own follow-up shows the numbers shift and the gap disappears on LSUN | Direction replicates; scope overstated |
| A4 | Quality and likelihood scale smoothly with compute (Abstract; Sec. 6; Fig. 10; Conclusion) | 4 model sizes on ImageNet-64; FID roughly a power law | Asymptotes hand-set; compute is theoretical FLOPs assuming full hardware utilization; NLL fit is poor; overfitting (App. F, G) | "Suggestive trend" is fair; "trivially" is overstated |
| A5 | Cosine noise schedule is better than linear (Sec. 3.2; Fig. 3-5; Tab. 1-2) | ImageNet-64 FID 31.3 to 27.0; NLL improves | CIFAR-10 FID gets *worse*. Linear is better at 256x256 (App. C). Dropout is a confound (0.1 vs 0.3) | Resolution-specific improvement, not a general one |

### A1. Claim: likelihood becomes competitive and sample quality is preserved

**What the paper shows.** Tab. 3 puts Improved DDPM at 2.94 bits/dim on CIFAR-10 and 3.53 bits/dim on ImageNet-64, ahead of Glow, Flow++, PixelCNN and the original DDPM.

**Weaknesses.**
- **The two halves of the claim come from different models.** The Tab. 3 numbers are the L_vlb models: CIFAR-10 NLL 2.94 / FID 11.47 (Tab. 2) and ImageNet-64 NLL 3.53 / FID 40.1 at 1.5M iterations (Tab. 1). The models the authors actually recommend (L_hybrid, Sec. 3.4) reach FID 3.19 and 19.2, but with NLL 3.17 and 3.57.
  - 3.17 bits/dim on CIFAR-10 is worse than PixelCNN (3.14), Flow++ (3.08) and every transformer or VAE row in Tab. 3.
  - 3.57 on ImageNet-64 ties PixelCNN.
  - The only single system that is good on both is the App. D ensemble (FID 19.9, NLL 3.52). It needs two separately trained networks.
- **"Competitive" does not mean leading.** Even the L_vlb numbers trail 5 of the 10 CIFAR-10 baselines in Tab. 3 (Sparse Transformer 2.80, PixelSNAIL 2.85, VDVAE 2.87, Image Transformer 2.90, NVAE 2.91). They trail 6 of the 10 ImageNet-64 baselines (Routing Transformer 3.43 through SPN, VDVAE and PixelSNAIL at 3.52). The Tab. 3 caption admits the transformer gap, so this is accurate but less impressive than the abstract suggests.
- **Much of the gain comes from a costly knob, not from the new ideas.** Raising the number of diffusion steps T from 1000 to 4000 accounts for a large share of the NLL improvement:

  | Dataset (source) | Raising T alone | Full hybrid recipe | T's share of the gain | T's share of the gain up to L_vlb |
  |---|---|---|---|---|
  | ImageNet-64 (Tab. 1) | 3.99 → 3.77 (0.22) | 3.99 → 3.62 (0.37) | ~59% | ~52% |
  | CIFAR-10 (Tab. 2) | 3.73 → 3.37 (0.36) | 3.73 → 3.17 (0.56) | ~64% | ~46% |

  Shares are [my calc]. A 4x longer chain also makes full sampling 4x slower, which is the problem Sec. 4 then has to fix.
- **The "baseline" is not untouched DDPM.**
  - The DDPM ImageNet-64 row in Tab. 3 (3.77) is the authors' own T=4000 run (Sec. 3). Ho et al. (2020) never reported ImageNet-64.
  - All runs, including the baseline, use a modified UNet: multi-head attention, attention at 8x8, and adaptive GroupNorm (App. A). The paper only says these changes improved FID slightly in preliminary experiments and gives no numbers. The architecture's effect is therefore mixed into every comparison.
- **The motivation is assumed rather than tested.**
  - Sec. 3 justifies chasing NLL with the belief that likelihood training forces a model to capture all modes.
  - The paper's own Fig. 2 shows the first few diffusion steps dominate the bound.
  - App. D (Fig. 13) says the L_vlb model concentrates on imperceptible details.
  - So lower bits/dim mostly buys better pixel-level modeling, not better coverage of what images depict (see Section B.1).
- **Overfitting is a confound.** All CIFAR-10 models overfit (App. F, Fig. 16). The cosine runs used dropout 0.3 and the linear runs 0.1 (App. A), so Tab. 2 compares different regularization as well as different schedules.

### A2. Claim: learned variances cut sampling steps about 10x at almost no quality cost

**What the paper shows.** Fig. 8 plots FID against 25, 50, 100, 200, 400, 1000 and 4000 sampling steps for ImageNet-64 and CIFAR-10. It compares mid-training and fully trained checkpoints. L_hybrid with learned variance stays flat down to about 100 steps, while the fixed-variance L_simple models get worse. App. B (Fig. 11) repeats this on LSUN bedrooms at 256x256.

**Weaknesses.**
- **The headline is inconsistent across the paper.**
  - The abstract says an order of magnitude fewer passes with negligible loss.
  - The introduction says as few as 50 forward passes.
  - Sec. 4 says 100 steps is enough for *near-optimal* FID on fully trained models.
  - The evidence exists only as plots, with no table values and no uncertainty.
- **Check which baseline "10x" refers to.** 100 steps is 10x fewer than Ho et al.'s T=1000, but 40x fewer than the authors' own 4000-step setting.
- **DDIM beats the method below 50 steps** on ImageNet-64, CIFAR-10 (Sec. 4) and LSUN (App. B).
  - DDIM was run with a different (constant) striding, because the paper's striding hurt it.
  - On LSUN the ranking also depends on training hyperparameters: DDIM prefers the smaller learning rate and batch size, while L_hybrid prefers the larger ones (App. B).
  - The authors' next paper (ADM, Sec. 2) switches to DDIM whenever it uses fewer than 50 steps.
- **Quality is measured only by FID.**
  - The unconditional ImageNet-64 FIDs use 10K samples.
  - Neither precision/recall nor human judgment is reported as steps decrease, so we cannot tell whether fewer steps cost fidelity or diversity.
  - Tab. 4 (the GAN comparison) uses 250 steps, so the fast-sampling regime and the coverage claim are never tested together.
- **Likelihood does not get faster.** In App. E (Fig. 15), strided evaluation drastically reduced NLL quality. The fix needs T/K extra evaluation steps.
- **No deployment metrics are reported.** The abstract calls the speed-up important for practical deployment. Yet the only timing information is that 4000 steps take *several minutes* on a modern GPU and that striding brings this down to *seconds* (Sec. 4). There is no latency, throughput, memory or energy measurement.
- **Rough compute per image.** App. A gives ~39 GFLOPs per forward pass for the 120M-parameter ImageNet-64 UNet. Per 64x64 image that is roughly [my calc]:

  | Sampling steps | Compute per image |
  |---|---|
  | 50 | 1.95 TFLOPs |
  | 100 | 3.9 TFLOPs |
  | 250 (used for Tab. 4, Fig. 9, Fig. 12) | 9.75 TFLOPs (using the 120M model's 39 GFLOPs; the Tab. 4 models are 100M and 270M, so the large model costs more) |
  | 4000 | 156 TFLOPs |

  A GAN needs one generator pass. The authors' follow-up paper concedes the gap. Diffusion models remain slower than GANs at sampling, and each forward pass alone takes 5-20x longer than a competing GAN generator (Dhariwal & Nichol 2021, Sec. 7).
- **Later work moved the frontier far past 50 steps** (useful context for "is 50 steps deployable?"):
  - DPM-Solver: CIFAR-10 FID 4.70 at 10 network evaluations and 2.87 at 20 (NeurIPS 2022).
  - Progressive distillation: FID 3.0 in 4 steps (ICLR 2022).
  - Consistency models: one-step FID 3.55 on CIFAR-10 and 6.20 on ImageNet-64 (ICML 2023).

### A3. Claim: diffusion reaches far higher recall than GANs at comparable FID

**What the paper shows.** Tab. 4 (class-conditional ImageNet-64, 50K samples, Inception-V3 features, K=5):

| Model | FID | Precision | Recall |
|---|---|---|---|
| BigGAN-deep | 4.06 | 0.86 | 0.59 |
| Improved Diffusion, small | 6.92 | 0.77 | 0.72 |
| Improved Diffusion, large | 2.92 | 0.82 | 0.71 |

The Conclusion upgrades this to: DDPMs match GAN sample quality with much better mode coverage.

**Weaknesses.**
- **The GAN baseline is weak and not compute-matched.**
  - It is a single BigGAN-deep, trained by the authors for 125K iterations, with 100M parameters across generator and discriminator, sampled without truncation (Tab. 4 caption; Sec. 5).
  - The diffusion "small" model trained for 1.7M steps. The "large" one has 270M parameters and trained for 250K iterations at batch size 2048 (Sec. 5; App. A).
  - No compute comparison is given. Each configuration was run once.
- **"Similar FID" is loose.** For the 100M-parameter models (diffusion small vs BigGAN-deep, both about 100M), FID is 6.92 vs 4.06, about 70% worse for diffusion. Where diffusion wins on both FID and recall (the large model), it loses on precision (0.82 vs 0.86).
- **The numbers depend on how they are computed.** The authors' own next paper re-scored the same two models (same FIDs, 4.06 and 2.92) and got different precision/recall (Dhariwal & Nichol 2021, Tab. 4):

  | Model | Recall here (Tab. 4) | Recall re-scored | Precision here | Precision re-scored |
  |---|---|---|---|---|
  | BigGAN-deep | 0.59 | 0.48 | 0.86 | 0.79 |
  | Improved Diffusion, large | 0.71 | 0.62 | 0.82 | 0.74 |

  The direction of the recall gap replicates (+0.12 here, +0.14 re-scored), but absolute values move by about 0.1, and ADM does not explain why [UNVERIFIED cause]. The paper also used K=5 rather than the default k=3 of Kynkäänniemi et al. (2019); larger k raises both precision and recall.
- **The advantage does not hold everywhere.**
  - On LSUN bedrooms, the authors' follow-up reports Improved DDPM recall 0.46 vs StyleGAN 0.48, with FID 4.24 vs 2.35.
  - At 256x256, the paper's own Tab. 5 has two-stage diffusion at FID 12.3 vs BigGAN-deep 7.0.
  - So "matching GAN quality" holds only at 64x64, and only against the self-trained BigGAN.
- **Recall is measured in an ImageNet-classifier feature space.** Kynkäänniemi et al. (ICLR 2023) show that FID in Inception space is so tied to ImageNet class probabilities that it can be cut substantially without visible improvement. They warn specifically about comparing sufficiently different architectures or training setups, which is exactly GAN vs diffusion. They also note that precision/recall rely on the same kind of features.
- **High recall can come from copying.** The FID reference set is the training set (Sec. 5; App. A; 50K training samples for LSUN), and the paper does not say precision/recall used anything else, so a model that reproduces training images can look like it has excellent coverage. Carlini et al. (2023) show this model family does reproduce training images (see C.1).
- **The metric itself is fragile.** Improved precision/recall is a k-nearest-neighbor estimate. Naeem et al. (ICML 2020) show it is not robust to outliers and can fail to recognize two identical distributions; they propose density and coverage instead.

### A4. Claim: quality and likelihood improve smoothly as model size and compute grow

**What the paper shows.**
- Four ImageNet-64 L_hybrid models with first-layer widths of 64, 96, 128 and 192 channels (30M, 68M, 120M and 270M parameters; Fig. 10).
- FID trend line: 4.00 + (2.5e-25 * C)^-0.22.
- NLL trend line: 3.40 + (3.0e-15 * C)^-0.17.

**Weaknesses.**
- **The base is small.** Four sizes, one dataset, one seed each. Each curve is a training trajectory, not a compute-optimal frontier. The x-axis spans roughly 10^17 to 10^20 FLOPs (read off Fig. 10).
- **The asymptotes are chosen by hand, not fitted.**
  - The FID constant (4.00) is approximated from the FID of real in-distribution data.
  - The NLL constant (3.40) is the state of the art, 3.43 (Routing Transformer), rounded down (Fig. 10 caption).
  - When the irreducible term is chosen by hand, the power-law form is largely set by that choice.
- **Compute is theoretical.** Footnote 3 says the x-axis assumes full hardware utilization, so it understates real GPU time and energy. The paper reports no GPU-hours anywhere.
- **NLL does not fit a power law cleanly** (Sec. 6). The authors offer an unexpectedly high irreducible loss or overfitting as possible reasons, without testing either.
- **Hyperparameters are not tuned per size.** The learning rate is scaled by 1/sqrt(channel multiplier) with no per-size sweep (Sec. 6). The FIDs are 10K-sample (App. A).
- **The Conclusion contradicts the body.** Sec. 6 calls the results preliminary, but the Conclusion says:

  > "more training compute trivially leads to better sample quality and log-likelihood"

  App. G reports that class-conditional ImageNet-64 FID got *worse* with longer training (overfitting). It concludes the best strategy is to stop early and spend extra compute on model size instead.

### A5. Claim: the cosine schedule beats the linear schedule

**What the paper shows.**
- Fig. 3-5 argue that the linear schedule wastes the last part of the forward process. Skipping up to 20% of the linear model's reverse process barely changes FID (Fig. 4).
- On ImageNet-64 at 200K iterations (Tab. 1), the cosine schedule improves both metrics:

  | Objective | FID (linear → cosine) | NLL (linear → cosine) |
  |---|---|---|
  | L_simple | 31.3 → 27.0 | 3.77 → 3.68 |
  | L_hybrid | 32.2 → 28.0 | 3.66 → 3.62 |

**Weaknesses.**
- **CIFAR-10 FID gets worse with cosine** (Tab. 2):

  | Objective | FID (linear → cosine) | NLL (linear → cosine) |
  |---|---|---|
  | L_simple | 2.90 → 3.05 | 3.37 → 3.26 |
  | L_hybrid | 3.07 → 3.19 | 3.26 → 3.17 |

  The best CIFAR-10 FID in the whole paper is the linear-schedule, T=4000, L_simple run (2.90).
- **Dropout is confounded.** CIFAR-10 used dropout 0.1 for linear and 0.3 for cosine, because cosine overfits faster (App. A; App. F, Fig. 16).
- **The schedule is tied to resolution.**
  - Sec. 3.2 already concedes the linear schedule is fine for high-resolution images, and App. C uses linear at 256x256 because it worked better there.
  - Later work formalizes this: the optimal schedule shifts toward more noise as image size grows (Chen 2023), and *simple diffusion* shifts the cosine schedule for high resolution (Hoogeboom et al., ICML 2023).
- **The shape and constants are hand-picked.** The authors call the cos^2 shape arbitrary (Sec. 3.2); the offset s = 0.008 is set by a rule of thumb (√β₀ slightly below one pixel bin, Sec. 3.2) rather than tuned. No alternative shapes were swept, and each comparison is a single run.
- **Descendants use it selectively.**
  - ADM uses cosine only for ImageNet-64 and linear for ImageNet-128/256/512 and LSUN (ADM, Tab. 11).
  - DALL-E 2 uses cosine up to its 64→256 upsampler and linear for 256→1024.
  - Stable Diffusion v1 uses a linear beta schedule (0.00085 to 0.012, 1000 steps).

### A6. Methodological issues that affect every claim

- **No variance estimates.** Every number is one training run. The only "seed" in the paper is fixed sampling noise for side-by-side pictures (Fig. 14, 30, 31). There are no confidence intervals, bootstrap estimates or repeated seeds.
- **Biased FID in the ablations.**
  - Unconditional ImageNet-64 ablations use 10K samples (App. A).
  - Footnote 2 says using more samples lowers FID by roughly 2 points, yet Tab. 1 differences are often about 1 point (27.0 vs 28.0; 31.3 vs 32.2).
  - The authors argue the bias is acceptable for relative comparisons. Chong & Forsyth (CVPR 2020) show finite-sample FID bias *differs between models*, so rankings at a fixed sample count can still be wrong.
- **Sample metrics use the training set as reference.** FID is computed against the training set (App. A; Sec. 5; full set for CIFAR-10 and ImageNet, 50K samples for LSUN), and the paper gives no separate reference for precision and recall. They therefore cannot penalize memorization. NLL, in contrast, is reported on test/validation data (Fig. 10, 16).
- **There are two different "ImageNet 64x64" datasets.**
  - Unconditional models use the official downsampled ImageNet-64.
  - Class-conditional models use center-cropped, area-downsampled images (App. A).
  - So Tab. 1 and 3 are not on the same data as Tab. 4.
- **Checkpoint selection is not described.** App. F says all CIFAR-10 models eventually overfit and reach similar best FIDs at some point; App. G recommends early stopping. The paper does not say how the checkpoints in Tab. 1-2 were chosen [UNVERIFIED whether by best FID]. If FID was used both to pick and to report, the numbers are optimistically biased.
- **No limitations, broader-impact or ethics section, and no model card** in the paper or the repository README, even though checkpoints were released publicly.

---

## B. "Train on one, test on another": pros and cons for this paper

The paper never evaluates a model outside its training distribution. It still leans on cross-dataset transfer in four places:
1. An ImageNet-trained Inception-V3 scores CIFAR-10, LSUN and ImageNet models.
2. Models trained with T=4000 steps are tested with K < T steps.
3. An upsampler trained on *real* 64x64 images is tested on *generated* 64x64 images (App. C).
4. Hyperparameters are transferred across model sizes (Sec. 6).

The most damaging test the paper did not run is this one: train on CIFAR-10, then measure likelihood on SVHN.

### B.1 Likelihood under distribution shift (the key evidence)

Each row trains a model on in-distribution (ID) data and tests on out-of-distribution (OOD) data. AUROC measures how well "ID or OOD?" is separated, where 0.5 is chance and 1.0 is perfect. **An AUROC far below 0.5 means the model systematically prefers the OOD data.**

| Study | Model trained on CIFAR-10 | Tested on | Result |
|---|---|---|---|
| Nalisnick et al., ICLR 2019 | Glow | SVHN test | Bits/dim: CIFAR-10 train 3.386, CIFAR-10 test 3.464, **SVHN test 2.389**, i.e. about 1 bit/dim *more* likely. Same effect for FashionMNIST→MNIST (2.958 vs 1.833) |
| Choi, Jang & Alemi 2018 (WAIC) | Glow | SVHN | Single-model log-density AUROC **7.5%**. A 5-model WAIC ensemble reaches 100%, but horizontally flipped CIFAR stays at **50.0%** |
| Serrà et al., ICLR 2020 | Glow / PixelCNN++ | SVHN | Negative log-likelihood AUROC **0.083 / 0.083**. A complexity-corrected score S (likelihood minus a compressor's code length) gives 0.950 / 0.929 |
| **Heng, Thiery & Soh, NeurIPS 2024 (DiffPath)** | **Public Improved-DDPM CIFAR-10 checkpoint**, NLL from the improved-diffusion code | SVHN, CelebA, CIFAR-100, Textures | NLL AUROC: SVHN **0.091**, CelebA 0.574, CIFAR-100 0.521, Textures 0.609. The reverse direction (SVHN-trained, CIFAR-10 as OOD) scores 0.990 |
| Goodier & Campbell, BMVC 2023 | DDPM with cosine schedule, L1 loss, T=1000 (denoising-diffusion-pytorch, not improved-diffusion) | SVHN | Per-timestep loss is lower for SVHN than for in-distribution CIFAR-10 (their Fig. 2), which is the complexity bias. Their whole-ELBO score still reaches AUROC 0.941; the corrected ratio (CCLR) peaks at 0.964. FashionMNIST→MNIST ELBO: 0.784 |
| Graham et al., CVPR Workshops 2023 | Glow likelihood baseline vs a DDPM noise-and-reconstruct method | SVHN | Glow likelihood AUC 6.1 vs DDPM reconstruction 97.9. The reconstruction method takes minutes per input, which the authors flag as a limitation for real-time use |
| Le Lan & Dinh, Entropy 2021 | Theory | n/a | Even a *perfect* density model cannot guarantee anomaly detection, because density values change under reparametrization. Likelihood-based detection rests on hidden assumptions |

**Why this hits the paper directly.**
- Sec. 3 motivates improving NLL as a proxy for capturing all the modes.
- The paper's Fig. 2 shows most of the bound comes from the first few, almost noise-free steps.
- App. D (Fig. 13) says the L_vlb model is better exactly at the start and end of the chain, focusing on imperceptible details.
- That is the same mechanism Serrà et al. and Goodier & Campbell identify: likelihood is dominated by local pixel complexity.
- So the evidence the paper offers for "better likelihood = better model of the data" does not survive a cross-dataset test. The model's actual released checkpoint fails it (AUROC 0.091).

**Deployment consequence.** Someone might use Improved-DDPM bits/dim as a novelty or quality filter, for example to flag unusual medical scans or to decide whether an input is "typical". That filter would rate simple, smooth images as the *most* typical, regardless of what they show.

**Caveat.** Goodier & Campbell's DDPM gives 0.941 on the same dataset pair, versus DiffPath's 0.091. They differ in codebase, loss and scoring: Goodier uses a weighted L1-loss surrogate, DiffPath computes the true bits/dim. The answer depends on *how* "likelihood" is computed, which is itself an argument for reporting OOD AUROC explicitly instead of assuming NLL means coverage.

### B.2 Scoring with an evaluator trained on a different dataset (Inception-V3 for FID and precision/recall)

- **Pros**
  - One fixed "ruler" lets results be compared across hundreds of papers.
  - It needs no labels for the target dataset and is cheap.
  - Inception features capture object-level semantics that pixel-level metrics miss.
- **Cons, with evidence**
  - **Resolution mismatch.** CIFAR-10 (32x32) and ImageNet-64 images are upsampled to Inception's 299x299 input. Parmar et al. (CVPR 2022) show resizing and compression choices noticeably shift FID.
  - **Domain mismatch.** LSUN bedrooms and CIFAR-10 are not ImageNet's label space.
  - **ImageNet-class leakage** (Kynkäänniemi et al., ICLR 2023):
    - Matching the Top-1 ImageNet-class histogram of generated images improves FID by about 10% (FFHQ 5.30 → 4.70).
    - Resampling to match "fringe" features cuts FFHQ FID 5.30 → 1.78 (-66.4%) and LSUN Places 12.96 → 3.59 (-72.3%).
    - FID computed in CLIP features barely moves under the same resampling (-4.3% and -11.1%).
    - The authors warn against comparing sufficiently different architectures with Inception-FID alone.
  - **Misalignment with human judgment.** Stein et al. (NeurIPS 2023) find no metric strongly correlates with human realism judgments. Diffusion models' human-judged realism is *not* reflected in FID, partly because of over-reliance on Inception-V3. They recommend DINOv2-ViT-L/14 features.
- **Special case: class-conditional ImageNet-64 (Tab. 4).** The evaluator was trained on ImageNet training images, the reference set is the ImageNet training set, and the generator was trained on ImageNet. All three share data. This favors models that reproduce ImageNet class statistics, and nothing in the pipeline can detect copying.

### B.3 Train on synthetic, test on real (a test the paper could have run)

- **Classification Accuracy Score (CAS)** (Ravuri & Vinyals, NeurIPS 2019): train a classifier only on generated images, then test it on real images. On ImageNet 256x256, top-1 accuracy was:

  | Training data | Top-1 on real ImageNet |
  |---|---|
  | Real images | 73.09% |
  | BigGAN-deep samples | 42.65% |
  | VQ-VAE-2 samples | 54.83% |

  The same paper reports that FID and Inception Score do not predict CAS.
- **GAN-train / GAN-test** (Shmelkov et al., ECCV 2018) splits this in two:
  - Train on generated, test on real ≈ recall/diversity.
  - Train on real, test on generated ≈ precision/quality.
- **The missed opportunity.** Improved DDPM cites Ravuri & Vinyals only to borrow VQ-VAE-2's FID (Tab. 5). It never computes CAS for its own class-conditional model, although CAS directly tests "better mode coverage" and per-class CAS would show which ImageNet classes are dropped.

### B.4 Train/test mismatches inside the paper

- **Trained with T=4000 steps, sampled with K steps** (Sec. 4).
  - The good: the learned variance rescales automatically for a shortened chain (Eq. 19). This is a genuine robustness property and the paper's best idea.
  - The bad: the likelihood does not transfer to fewer steps (App. E).
- **Trained on real low-resolution inputs, tested on generated ones** (App. C). The 64→256 upsampler is trained on real downsampled images but applied to generated 64x64 samples; FID goes from 2.92 at 64x64 to 12.3 at 256x256. Cascaded Diffusion Models (Ho et al., JMLR 2022) later found cascade quality relies crucially on conditioning augmentation that prevents compounding error at sampling time.
- **Schedules tuned at 32-64 px do not transfer to 256 px** (App. C; see A5).
- **Deepfake detectors trained on GAN images, tested on diffusion images** (Ricker et al., VISAPP 2024). This includes images from Improved DDPM's LSUN bedroom model.
  - AUROC of pre-trained GAN-image detectors drops by 15.2% on average for diffusion images compared with GAN images.
  - A detector trained on all GANs reaches only an average Pd@1%FAR of 26.34% on diffusion images (the detection rate when only 1% of real images may be falsely flagged).
  - On IDDPM images specifically, the pre-trained detectors range from AUROC 62.9 to 94.3 and Pd@1%FAR 1.3% to 25.7%.

### B.5 Pros and cons of cross-dataset evaluation for this paper

| Pros | Cons |
|---|---|
| Exposes shortcuts that in-distribution NLL hides (CIFAR-10→SVHN AUROC 0.091) | Results depend on direction (CIFAR→SVHN 0.091 vs SVHN→CIFAR 0.990) and on the dataset pair (CelebA 0.574), so cherry-picking is possible |
| Mirrors deployment, where inputs drift away from the training data | The evaluator's own domain gap (ImageNet classes; upsampling 32→299) gets mixed into the score |
| A fixed evaluator makes results comparable across models and papers | When the evaluator and generator share training data (ImageNet), the score is inflated and cannot detect copying |
| Synthetic→real tests (CAS) measure downstream usefulness, not only realism | Hard to attribute a failure to the model, the metric or dataset statistics (Le Lan & Dinh) |
| Cheap: needs no new labels | Sensitive to the scoring choice (0.091 vs 0.941 on the same pair) |
| Shows whether design choices (schedule, step count) transfer to new conditions | Some cross-dataset tests are expensive (DDPM reconstruction OOD takes minutes per input) |

---

## C. Deployment and sociotechnical issues, with concrete evidence

### C.1 Memorization and privacy

**This exact model family memorizes training data** (Carlini et al., USENIX Security 2023):
- **Released checkpoint.** From 1M unconditional samples of the public **Improved-DDPM CIFAR-10 checkpoint**, they extracted **301** training images (FID 2.9).
  - Original DDPM: 232 images (FID 3.2).
  - StyleGAN-ADA: 150 images at the same FID of 2.9, so the diffusion model memorized about twice as much at equal quality.
- **Models trained with the improved-diffusion code.** They trained 16 models (each on half of CIFAR-10) with OpenAI's improved-diffusion repository and generated 2^20 samples. They found **1,280 unique extracted training images, 2.5% of the dataset**.
- **Better models leak more.** As FID improved over training, membership-inference success rose from 7% to nearly 100% TPR at 1% FPR (the share of training images correctly identified when only 1% of non-training images are falsely flagged).
- **Diffusion vs GANs.** Diffusion models showed about 50% TPR at 0.1% FPR, versus under 30% for GANs. The authors conclude diffusion models are less private than GANs.

**Link to the paper.**
- App. F and G report overfitting but treat it only as an FID problem to be solved by early stopping.
- FID against the training set cannot detect memorization; it *rewards* it.
- The scaling recommendation (bigger models, early stopping) is exactly the regime Carlini et al. found to leak more.

### C.2 Misuse, deepfakes and provenance

- **The paper itself** has no misuse discussion. The repository releases checkpoints under MIT with no model card.
- **The direct descendant shows the risk was recognized.**
  - GLIDE builds on this line of work (the ADM architecture plus this paper's learned variances).
  - OpenAI released only a 300M-parameter model trained on a *filtered* dataset, and removed images containing people.
  - Their stated reason: releasing without safeguards would significantly lower the skill needed to create convincing disinformation or deepfakes. They also filtered for violent images and hate symbols.
- **Detection lags** (see B.4). Detectors tuned on GAN images lose 15.2% AUROC on average on diffusion images, including Improved DDPM samples (Ricker et al., VISAPP 2024).

### C.3 Energy, latency and cost per image

- **What the paper gives.**
  - Several minutes per sample at 4000 steps (Sec. 4).
  - About 39 GFLOPs per forward pass (App. A).
  - A scaling x-axis in theoretical FLOPs assuming full hardware utilization (footnote 3).
  - No GPU-hours, kWh or CO2 figures.
- **Latency.** Each step is a full UNet pass, so cost grows linearly with step count. Using the compute figures from A2: 50 steps ≈ 2 TFLOPs, 250 steps ≈ 10 TFLOPs per 64x64 image [my calc]. The authors' follow-up says each pass is 5-20x slower than a GAN generator (ADM, Sec. 7).
- **What deployed descendants cost** (Luccioni, Jernite & Strubell, FAccT 2024; measured on modern text-to-image pipelines, *not* on IDDPM itself):
  - Image generation is the most energy-hungry task studied: a mean of **2.907 kWh per 1,000 inferences**, more than 60x text generation and about 1,450x text classification (0.002 kWh) [ratio is my calc].
  - The least efficient image model used **11.49 kWh per 1,000 images**, about 522 smartphone charges, or roughly half a charge per image.
  - The most carbon-intensive model (stable-diffusion-xl-base-1.0) emitted **1,594 g CO2 per 1,000 images**, about 4.1 miles driven by an average gasoline car.
- **Why it matters for this paper.** Iterative sampling, the cost Sec. 4 tried to reduce, is the main reason image generation dominates these measurements. Reporting "FID vs steps" without "FID vs joules" hides the trade-off deployers actually face.

### C.4 Dataset provenance and representational bias

- **CIFAR-10** is a labeled subset of the **80 Million Tiny Images** dataset (per the CIFAR website).
  - MIT formally withdrew Tiny Images on 29 June 2020, after Prabhu & Birhane documented derogatory category labels and offensive images.
  - MIT said the dataset was too large, and its 32x32 images too small, for manual inspection to guarantee removal of offensive content.
  - CIFAR-10's 10 classes are benign, but its provenance (web-scraped, no consent) is the same.
- **ImageNet (ILSVRC)** (Yang et al., ICML 2022): 243,198 of 1,431,093 annotated ILSVRC images (**17%**) contain at least one face. These are incidental people who never consented.
  - In the full ImageNet "person" subtree, 1,593 of 2,832 categories were judged unsafe or offensive and only 158 remained safe and imageable (Yang et al., FAT* 2020).
  - Only 3 of the 1,000 ILSVRC classes are person categories, so the main risk for this paper is incidental faces, not offensive labels.
- **LSUN bedrooms** are web images collected with a humans-in-the-loop pipeline (Yu et al. 2015), with no consent from the people whose homes appear in them.
- **The sociotechnical point.** "High recall" means faithfully reproducing the training distribution, *including its skews*. Recall against a biased reference set rewards reproducing that bias. Nothing in the paper measures who or what is over- or under-represented in samples.

### C.5 Copyright and consent

- Somepalli et al. (CVPR 2023):
  - About 1.88% of random Stable Diffusion v1.4 generations had similarity of 0.5 or more to images in a 12M-image LAION subset.
  - Diffusion models trained on *small* datasets copy blatantly; their CelebA models trained on 300 or 3,000 images reproduced training images.
  - An ImageNet-trained latent diffusion model showed no significant copying.
  - So risk rises as datasets shrink, which is relevant to anyone fine-tuning this code on a small private or artistic dataset.
- Carlini et al. (2023) extracted over a hundred near-identical training replicas from Stable Diffusion and Imagen, including photos of identifiable people and trademarked logos.
- Downstream data harms in this lineage: a Stanford Internet Observatory report (Thiel, Dec 2023) found many hundreds of instances of known child sexual abuse material in LAION-5B, parts of which trained Stable Diffusion. This is a lineage issue (code and ideas), not this paper's data.

### C.6 Downstream impact: what this paper's ideas became

| System | What it inherits from Improved DDPM | Evidence |
|---|---|---|
| ADM, "Diffusion Models Beat GANs" (Dhariwal & Nichol, NeurIPS 2021) | Uses the hybrid objective and learned-variance parameterization throughout. Switches to DDIM below 50 steps because IDDPM found that beneficial. Cosine schedule for ImageNet-64 only; linear elsewhere | ADM Sec. 2; Tab. 11 |
| GLIDE (Nichol et al., ICML 2022) | ADM architecture; explicitly adopts this paper's learned-sigma technique for training its models | GLIDE paper |
| DALL-E 2 / unCLIP (Ramesh et al. 2022) | Decoder is the 3.5B GLIDE model with the same diffusion hyperparameters. Trained with learned sigma and sampled with 250 strided steps as in Nichol & Dhariwal 2021. Cosine schedule for the prior, decoder and 64→256 upsampler; linear for 256→1024 | unCLIP Sec. 2.1 and hyperparameter table |
| Stable Diffusion / latent diffusion (Rombach et al., CVPR 2022) | **Code lineage only.** `ldm/models/diffusion/ddpm.py` describes itself as a mixture of lucidrains' denoising-diffusion-pytorch, openai/improved-diffusion and taming-transformers. The README says the code builds heavily on OpenAI's ADM codebase. But SD v1 uses the *linear* schedule (0.00085 to 0.012), epsilon prediction and fixed variance (`learn_logvar=False`) | GitHub sources |

**Takeaway.** The paper's choices, judged only by FID/NLL on CIFAR-10 and ImageNet-64, became defaults inside systems used by millions. The risks those systems raise (memorization, deepfakes, energy, bias) were never part of the original evaluation.

---

## D. Proposed "beyond FID" sociotechnical metrics and tools

Each row is something you could run on the released Improved-DDPM checkpoints.

| # | Metric / tool | What it catches that FID and NLL miss | How to run it on Improved DDPM | Reference |
|---|---|---|---|---|
| D1 | **Memorization and generalization audit**: extraction rate, membership-inference TPR@low-FPR, train-vs-held-out FID gap | Copying, privacy and copyright risk. FID against the training set *rewards* copying | (a) Generate 2^20 samples. Flag near-copies with the calibrated nearest-neighbor ℓ2 rule (alpha=0.5, n=50; Carlini) or SSCD/DINO similarity ≥ 0.5 (Somepalli). Report the % of the training set extracted. (b) LiRA/loss-threshold membership inference, reported as TPR@0.1%FPR. (c) FID against the *test* set minus FID against the *train* set. (d) The C_T data-copying test (Meehan et al.) or "authenticity" (Alaa et al.) | Carlini 2023; Somepalli 2023; Meehan 2020; Alaa 2022 |
| D2 | **OOD AUROC (the "train on one, test on another" check)** | Whether better bits/dim actually reflects knowing the data distribution. The paper's Sec. 3 motivation gets checked directly | Train on CIFAR-10. Score CIFAR-10 test vs SVHN, CelebA, CIFAR-100 and Textures with (i) raw bits/dim, (ii) a complexity-corrected score (Serrà's S; Goodier's CCLR) and (iii) a non-likelihood score (DiffPath). Report AUROC and FPR@95%TPR in both directions | Nalisnick 2019; Serrà 2020; Heng 2024; Goodier 2023 |
| D3 | **Cost per image**: latency, throughput, energy, carbon; quality-vs-cost Pareto curve | The deployment claim the abstract makes but never measures | Report network evaluations x FLOPs per pass. Measure wall-clock latency at batch size 1 and throughput at batch size 64 on a named GPU. Measure kWh per 1,000 images with CodeCarbon and convert to gCO2 with the local grid intensity. Plot **FID vs kWh** instead of FID vs steps, include DDIM and DPM-Solver, and report training GPU-hours | Luccioni 2024; Lacoste 2019 |
| D4 | **Downstream utility and coverage *per class or subgroup***: CAS, per-class recall/coverage, fairness discrepancy | Whether "mode coverage" holds for *every* class, or the mean hides dropped classes and groups | (a) CAS: train a ResNet-50 on generated class-conditional ImageNet-64 samples and test on real validation images; report top-1/top-5 and the **worst-10 per-class accuracies**. (b) Per-class density and coverage (Naeem), reported as a distribution (min, quartiles), not a mean. (c) For data with people, the gap in perceived-attribute distribution vs the reference set, with error-corrected measurement (CLEAM) because attribute classifiers are noisy | Ravuri & Vinyals 2019; Shmelkov 2018; Naeem 2020; Teo 2023; Choi 2020 |
| D5 | **Human-aligned, ImageNet-independent quality with uncertainty** | Inception/ImageNet bias; single-run noise | FD computed on DINOv2-ViT-L/14 features (Stein) and CMMD on CLIP features (Jayasumana), plus KID (unbiased estimator, usable at 10K samples). Run at least 3 seeds and report bootstrap confidence intervals. Add a small human study (HYPE-style timed realism, or two-alternative forced choice) | Stein 2023; Jayasumana 2024; Binkowski 2018; Zhou 2019 |

**Optional sixth metric: detectability and provenance.** Report the AUROC and Pd@1%FAR of off-the-shelf synthetic-image detectors on the model's samples (Ricker et al.). If sampling is distributed, also test whether watermarks survive.

**Summary: an evaluation card for generative models.**
- *Quality* (FID-DINOv2 with CI)
- *Coverage per class* (CAS worst-10)
- *Privacy* (extraction %)
- *Robustness* (OOD AUROC)
- *Cost* (kWh and ms per image)
- *Provenance* (detector AUROC)

---

## Claims I could not verify

- **Exact values on Fig. 8 and Fig. 10.** The paper gives these only as plots. Do not quote precise FIDs at 50 or 100 steps.
- **Why ADM's precision/recall differ from Tab. 4.** ADM gives no explanation (it says only that it re-computes all metrics with one codebase and the full training set as reference). The direction of the change is verified; the cause is not.
- **How the Tab. 1-2 checkpoints were selected.** The paper does not say.
- **Goodier & Campbell's intermediate CCLR values.** The PDF table did not extract cleanly. Only ELBO AUROC 0.941, peak CCLR 0.964 and FashionMNIST→MNIST ELBO 0.784 were confirmed.
- **Ricker et al.'s per-detector column labels.** The table header was garbled. Only the aggregates (15.2% average AUROC drop; Pd@1%FAR of 26.34% for the GAN-trained detector on diffusion images) and the IDDPM ranges (AUROC 62.9-94.3, Pd@1%FAR 1.3-25.7%) are used.
- **Exact CSAM counts in LAION-5B** (3,226 suspected / 1,008 validated). These appear only in secondary coverage; the primary Stanford page describes many hundreds of known instances, so only that wording is used.
- **Venues confirmed from memory or secondary sources** rather than the arXiv page: Carlini (USENIX Security 2023), Parmar (CVPR 2022), Choi et al. 2020 (ICML 2020), Alaa (ICML 2022), CMMD (CVPR 2024), HYPE (NeurIPS 2019), Cascaded Diffusion Models (JMLR 2022), simple diffusion (ICML 2023). The arXiv IDs below are correct.

---

## References

**Primary paper**
1. Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. ICML 2021, PMLR 139:8162-8171. https://proceedings.mlr.press/v139/nichol21a.html | https://arxiv.org/abs/2102.09672 | Code: https://github.com/openai/improved-diffusion

**Likelihood and OOD ("train on one, test on another")**

2. Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., & Lakshminarayanan, B. (2019). Do Deep Generative Models Know What They Don't Know? ICLR 2019. https://arxiv.org/abs/1810.09136
3. Choi, H., Jang, E., & Alemi, A. A. (2018). WAIC, but Why? Generative Ensembles for Robust Anomaly Detection. https://arxiv.org/abs/1810.01392
4. Serrà, J., Álvarez, D., Gómez, V., Slizovskaia, O., Núñez, J. F., & Luque, J. (2020). Input Complexity and Out-of-Distribution Detection with Likelihood-Based Generative Models. ICLR 2020. https://arxiv.org/abs/1909.11480
5. Le Lan, C., & Dinh, L. (2021). Perfect Density Models Cannot Guarantee Anomaly Detection. Entropy 23(12):1690. https://arxiv.org/abs/2012.03808 | https://doi.org/10.3390/e23121690
6. Goodier, J., & Campbell, N. D. F. (2023). Likelihood-based Out-of-Distribution Detection with Denoising Diffusion Probabilistic Models. BMVC 2023. https://arxiv.org/abs/2310.17432
7. Graham, M. S., Pinaya, W. H. L., Tudosiu, P.-D., Nachev, P., Ourselin, S., & Cardoso, M. J. (2023). Denoising Diffusion Models for Out-of-Distribution Detection. CVPR Workshops (VAND) 2023. https://arxiv.org/abs/2211.07740 | https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.html
8. Heng, A., Thiery, A. H., & Soh, H. (2024). Out-of-Distribution Detection with a Single Unconditional Diffusion Model (DiffPath). NeurIPS 2024. https://arxiv.org/abs/2405.11881
9. Theis, L., van den Oord, A., & Bethge, M. (2016). A Note on the Evaluation of Generative Models. ICLR 2016. https://arxiv.org/abs/1511.01844

**Evaluation metrics**

10. Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019). Improved Precision and Recall Metric for Assessing Generative Models. NeurIPS 2019. https://arxiv.org/abs/1904.06991
11. Kynkäänniemi, T., Karras, T., Aittala, M., Aila, T., & Lehtinen, J. (2023). The Role of ImageNet Classes in Fréchet Inception Distance. ICLR 2023. https://arxiv.org/abs/2203.06026
12. Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., & Yoo, J. (2020). Reliable Fidelity and Diversity Metrics for Generative Models. ICML 2020. https://arxiv.org/abs/2002.09797
13. Stein, G., et al. (2023). Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models. NeurIPS 2023. https://arxiv.org/abs/2306.04675
14. Chong, M. J., & Forsyth, D. (2020). Effectively Unbiased FID and Inception Score and Where to Find Them. CVPR 2020. https://arxiv.org/abs/1911.07023
15. Parmar, G., Zhang, R., & Zhu, J.-Y. (2022). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. CVPR 2022. https://arxiv.org/abs/2104.11222
16. Ravuri, S., & Vinyals, O. (2019). Classification Accuracy Score for Conditional Generative Models. NeurIPS 2019. https://arxiv.org/abs/1905.10887
17. Shmelkov, K., Schmid, C., & Alahari, K. (2018). How Good Is My GAN? ECCV 2018. https://arxiv.org/abs/1807.09499
18. Bińkowski, M., Sutherland, D. J., Arbel, M., & Gretton, A. (2018). Demystifying MMD GANs (Kernel Inception Distance). ICLR 2018. https://arxiv.org/abs/1801.01401
19. Jayasumana, S., et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD). CVPR 2024. https://arxiv.org/abs/2401.09603
20. Zhou, S., Gordon, M. L., Krishna, R., Narcomey, A., Fei-Fei, L., & Bernstein, M. S. (2019). HYPE: A Benchmark for Human eYe Perceptual Evaluation of Generative Models. NeurIPS 2019. https://arxiv.org/abs/1904.01121
21. Alaa, A. M., van Breugel, B., Saveliev, E., & van der Schaar, M. (2022). How Faithful Is Your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models. ICML 2022. https://arxiv.org/abs/2102.08921
22. Meehan, C., Chaudhuri, K., & Dasgupta, S. (2020). A Non-Parametric Test to Detect Data-Copying in Generative Models. AISTATS 2020. https://arxiv.org/abs/2004.05675
23. Teo, C. T. H., Abdollahzadeh, M., & Cheung, N.-M. (2023). On Measuring Fairness in Generative Models (CLEAM). NeurIPS 2023. https://arxiv.org/abs/2310.19297
24. Choi, K., Grover, A., Singh, T., Shu, R., & Ermon, S. (2020). Fair Generative Modeling via Weak Supervision. ICML 2020. https://arxiv.org/abs/1910.12008

**Memorization, privacy, copyright, misuse**

25. Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., Balle, B., Ippolito, D., & Wallace, E. (2023). Extracting Training Data from Diffusion Models. USENIX Security 2023. https://arxiv.org/abs/2301.13188
26. Somepalli, G., Singla, V., Goldblum, M., Geiping, J., & Goldstein, T. (2023). Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models. CVPR 2023. https://arxiv.org/abs/2212.03860
27. Ricker, J., Damm, S., Holz, T., & Fischer, A. (2024). Towards the Detection of Diffusion Model Deepfakes. VISAPP 2024 (extended arXiv version). https://arxiv.org/abs/2210.14571
28. Thiel, D. (2023). Identifying and Eliminating CSAM in Generative ML Training Data and Models. Stanford Internet Observatory. https://purl.stanford.edu/kh752sm9123

**Energy and cost**

29. Luccioni, A. S., Jernite, Y., & Strubell, E. (2024). Power Hungry Processing: Watts Driving the Cost of AI Deployment? ACM FAccT 2024. https://arxiv.org/abs/2311.16863 | https://doi.org/10.1145/3630106.3658542
30. Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. (2019). Quantifying the Carbon Emissions of Machine Learning. https://arxiv.org/abs/1910.09700

**Datasets and bias**

31. Krizhevsky, A. CIFAR-10 and CIFAR-100 datasets (described as labeled subsets of 80 Million Tiny Images). https://www.cs.toronto.edu/~kriz/cifar.html
32. Torralba, A., Fergus, R., & Freeman, B. (2020). 80 Million Tiny Images withdrawal statement (29 June 2020). https://groups.csail.mit.edu/vision/TinyImages/ | Coverage: https://www.deeplearning.ai/the-batch/tiny-images-outsized-biases/
33. Prabhu, V. U., & Birhane, A. (2020). Large Image Datasets: A Pyrrhic Win for Computer Vision? https://arxiv.org/abs/2006.16923
34. Yang, K., Qinami, K., Fei-Fei, L., Deng, J., & Russakovsky, O. (2020). Towards Fairer Datasets: Filtering and Balancing the Distribution of the People Subtree in the ImageNet Hierarchy. FAT* 2020. https://arxiv.org/abs/1912.07726
35. Yang, K., Yau, J., Fei-Fei, L., Deng, J., & Russakovsky, O. (2022). A Study of Face Obfuscation in ImageNet. ICML 2022. https://arxiv.org/abs/2103.06191

**Lineage and follow-up diffusion work**

36. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020. https://arxiv.org/abs/2006.11239
37. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. ICLR 2021. https://arxiv.org/abs/2010.02502
38. Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis (ADM). NeurIPS 2021. https://arxiv.org/abs/2105.05233
39. Nichol, A., et al. (2022). GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. ICML 2022. https://arxiv.org/abs/2112.10741
40. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2 / unCLIP). https://arxiv.org/abs/2204.06125
41. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022. https://arxiv.org/abs/2112.10752 | https://github.com/CompVis/latent-diffusion | https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py | https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml
42. Ho, J., Saharia, C., Chan, W., Fleet, D. J., Norouzi, M., & Salimans, T. (2022). Cascaded Diffusion Models for High Fidelity Image Generation. JMLR 23. https://arxiv.org/abs/2106.15282
43. Chen, T. (2023). On the Importance of Noise Scheduling for Diffusion Models. https://arxiv.org/abs/2301.10972
44. Hoogeboom, E., Heek, J., & Salimans, T. (2023). simple diffusion: End-to-end Diffusion for High Resolution Images. ICML 2023. https://arxiv.org/abs/2301.11093
45. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. NeurIPS 2022. https://arxiv.org/abs/2206.00927
46. Salimans, T., & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. ICLR 2022. https://arxiv.org/abs/2202.00512
47. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency Models. ICML 2023. https://arxiv.org/abs/2303.01469
