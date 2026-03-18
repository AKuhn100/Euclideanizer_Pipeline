# Euclideanizer Scoring Specification

*Per-run evaluation: one seed × one DistMap × one Euclideanizer*

---

## 1. Overview

Scoring evaluates Euclideanizer outputs only (DistMap is fixed). Each component score is in [0, 1] with 1 = best. Missing components (skipped or unsaved analysis blocks) are omitted from the aggregate; resaving additional blocks and rescoring will include them.

---

## 2. Normalization and Aggregation

### 2.1 Data normalization (z-score)

Before computing any discrepancy metric, normalize both sides of the comparison using the combined pool of both distributions:

```
x_norm = (x − μ) / σ

```

where μ and σ are computed over the **combined pool** (e.g. exp + recon together, not each side separately). This puts all metrics — Rg, RMSD, Q, pairwise distances, curve values — into standard deviation units, making discrepancies comparable across all components.

Use z-scoring rather than min/max normalization. Min/max is sensitive to outlier structures that collapse the scale; z-scoring is robust to these.

### 2.2 Exponential kernel scoring (τ = 1)

All components (except Recon RMSD, Recon Q, and Clustering — see §4.4, §4.7, §4.8) use:

```
score = e^{-d}

```

where `d` is the discrepancy on z-scored data and τ = 1. Because all discrepancies are in the same σ-units after z-scoring, a single τ = 1 applies everywhere without per-metric tuning.

**Interpretation:** a discrepancy of 1σ gives score ≈ 0.37; 0σ gives score = 1.

> **Note on Wasserstein sample noise:** Wasserstein-1 between finite samples of the same distribution is not exactly 0. Estimate this noise floor empirically (compute W1 between two subsamples of the same set) and confirm it is well below 1σ for your typical sample sizes.

### 2.3 Aggregation: geometric mean

Overall score = geometric mean of component scores **only when all components are present** (no missing data). If any component is missing, `overall_score` is set to NaN so that composite scores are comparable across runs (e.g. one run with 8 categories vs another with 5 would otherwise be incomparable).

```
s̄ = ( ∏ᵢ sᵢ )^(1/n)   when all components present; else overall_score = NaN
```

The geometric mean ensures no single very bad component can be masked by strong performance elsewhere.

---

## 3. Metrics

Two metrics are used everywhere they apply, after z-score normalization of the input data:

- **MAE** — for curve and matrix comparisons (scaling P(s), average distance map). On-scale, robust to outliers, consistent.
- **Wasserstein-1 (W1)** — for distribution comparisons (Rg, RMSD, Q, pairwise distances). Well-defined on empirical samples, symmetric, no shared-support requirement. Preferred over KL divergence.

Two special cases use ratio-based discrepancies rather than Wasserstein or MAE:

- **Recon RMSD** — scored relative to the test→train RMSD baseline (§4.4).
- **Recon Q** — scored relative to the test→train Q baseline (§4.7).

---

## 4. Component Definitions

### 4.1 Reconstruction (train and test independently)

**Goal:** Reconstructed structures should match experimental structures in scaling, Rg, pairwise distances, and average distance map — evaluated separately on train and test sets.

- **Scaling:** MAE between P_exp(s) and P_recon(s) after z-scoring both curves.
- **Rg:** Wasserstein between Rg_exp and Rg_recon samples after z-scoring the combined pool.
- **Pairwise distances:** mean over **all genomic lags** s = 1,…,N−1 of Wasserstein between d(i,i+s) distributions (z-scored). Experimental and reconstructed samples use the **same** train/test structure counts (`plotting.max_train` / `max_test`; `null` = full split). Plotting may show only a subset of lags (e.g. 20) for display; scoring uses all lags.
- **Average map:** MAE between `<D>_exp` and `<D>_recon` matrices after z-scoring.

8 components total (4 measures × train/test).

---

### 4.2 Generation (gen vs composite experimental)

**Goal:** Generated structures (variance = 1) should match the composite (train + test) experimental distribution in Rg, scaling, pairwise distances, and average map.

Same four measures as §4.1, comparing gen vs composite exp.

4 components total.

---

### 4.3 Gen RMSD

**Goal:** The min-RMSD distributions of gen→train and gen→test should each match test→train (the no-model reference distribution).

- Gen→train RMSD vs test→train RMSD: Wasserstein on z-scored combined pool.
- Gen→test RMSD vs test→train RMSD: Wasserstein on z-scored combined pool.

2 components total.

---

### 4.4 Recon RMSD

**Goal:** Recon RMSD distributions should be shifted toward 0 relative to test→train. The ideal is RMSD = 0; test→train is the no-model baseline.

```
d = median(recon RMSD) / median(test→train RMSD)

```

- d → 0 as recon approaches perfect (score → 1).
- d = 1 when recon is no better than the test→train baseline (score ≈ 0.37).
- d > 1 when recon is worse than baseline (score < 0.37).

No manual threshold (Δ_ref) needed. 2 components total (train recon, test recon).

---

### 4.5 Latent (train vs test identity)

**Goal:** The latent distribution should be identical for train and test per dimension — both mean and std vectors should match.

- **Means:** MAE between μ_train and μ_test vectors, after z-scoring across dimensions.
- **Stds:** MAE between σ_train and σ_test vectors, after z-scoring across dimensions.

Z-scoring across dimensions prevents a single high-variance dimension from dominating. 2 components total.

---

### 4.6 Gen Q

**Goal:** The max-Q distributions of gen→train and gen→test should each match test→train.

- Gen→train Q vs test→train Q: Wasserstein on z-scored combined pool.
- Gen→test Q vs test→train Q: Wasserstein on z-scored combined pool.

2 components total.

---

### 4.7 Recon Q

**Goal:** Recon Q distributions should be shifted toward 1 relative to test→train. The ideal is Q = 1; test→train is the no-model baseline.

```
d = (1 − median(recon Q)) / (1 − median(test→train Q))

```

- d → 0 as recon Q approaches 1 (score → 1).
- d = 1 when recon Q equals the test→train baseline (score ≈ 0.37).
- d > 1 when recon is worse than baseline (score < 0.37).

No manual threshold needed. 2 components total (train recon, test recon).

---

### 4.8 Clustering (mixing)

**Goal:** Generated structures should mix with both train and test experimental structures; reconstructed structures should mix with their corresponding experimental set.

```
ratio = observed mixing / expected mixing (random baseline)
d = max(0, 1 − ratio)

```

- ratio ≥ 1: mixing at least as good as random → d = 0 → score = 1.
- ratio < 1: worse than random → d > 0 → score < 1.

> **Note:** the expected mixing baseline must account for unequal cluster sizes.

8 components total: coord and distmap for each of gen+train, gen+test, recon train, recon test.

---

## 5. Summary Table


| Category   | # Components | Primary metric                         | Discrepancy d                                      |
| ---------- | ------------ | -------------------------------------- | -------------------------------------------------- |
| Recon      | 8            | MAE (curves/maps), W1 (distributions) | MAE or Wasserstein on z-scored data                |
| Gen        | 4            | MAE (curves/maps), W1 (distributions) | MAE or Wasserstein on z-scored data                |
| Gen RMSD   | 2            | W1                                    | Wasserstein on z-scored samples                    |
| Recon RMSD | 2            | Ratio to baseline                      | median(recon) / median(test→train)                 |
| Latent     | 2            | MAE (per-dim means, stds)              | MAE on z-scored vectors                            |
| Gen Q      | 2            | W1                                    | Wasserstein on z-scored samples                    |
| Recon Q    | 2            | Ratio to baseline                      | (1 − median(recon Q)) / (1 − median(test→train Q)) |
| Clustering | 8            | Observed/expected mixing ratio         | max(0, 1 − ratio)                                  |


**Total: 30 components** when all data are present. Overall score = geometric mean of component scores only when all are present; otherwise NaN. This keeps composite scores comparable across runs.

---

## 6. Full Component Table


| Category   | Component               | Data                               | Metric / d                                         | Score  |
| ---------- | ----------------------- | ---------------------------------- | -------------------------------------------------- | ------ |
| Recon      | Scaling (train)         | P_exp_train(s) vs P_recon_train(s) | MAE on z-scored curves                             | e^{-d} |
|            | Scaling (test)          | P_exp_test(s) vs P_recon_test(s)   | MAE on z-scored curves                             | e^{-d} |
|            | Rg (train)              | Rg exp vs recon (train)            | Wasserstein on z-scored samples                    | e^{-d} |
|            | Rg (test)               | Rg exp vs recon (test)             | Wasserstein on z-scored samples                    | e^{-d} |
|            | Pairwise dists (train)  | d(i,i+s) exp vs recon (train)      | mean_s W1_s, z-scored                             | e^{-d} |
|            | Pairwise dists (test)   | d(i,i+s) exp vs recon (test)       | mean_s W1_s, z-scored                             | e^{-d} |
|            | Avg map (train)         | <D>_exp_train vs <D>_recon_train   | MAE on z-scored matrices                           | e^{-d} |
|            | Avg map (test)          | <D>_exp_test vs <D>_recon_test     | MAE on z-scored matrices                           | e^{-d} |
| Gen        | Rg                      | Rg gen vs Rg composite exp         | Wasserstein on z-scored samples                    | e^{-d} |
|            | Scaling                 | P_gen(s) vs P_exp_composite(s)     | MAE on z-scored curves                             | e^{-d} |
|            | Pairwise dists          | Per-s Wasserstein, then mean       | mean_s W1_s, z-scored                             | e^{-d} |
|            | Avg map                 | <D>_gen vs <D>_exp_composite       | MAE on z-scored matrices                           | e^{-d} |
| Gen RMSD   | Gen→train vs test→train | Min RMSD distributions             | Wasserstein on z-scored samples                    | e^{-d} |
|            | Gen→test vs test→train  | Min RMSD distributions             | Wasserstein on z-scored samples                    | e^{-d} |
| Recon RMSD | Train recon             | Recon train & test→train medians   | median(recon) / median(test→train)                 | e^{-d} |
|            | Test recon              | Recon test & test→train medians    | median(recon) / median(test→train)                 | e^{-d} |
| Latent     | Means (train vs test)   | Per-dim means μ_train, μ_test      | MAE on z-scored means                              | e^{-d} |
|            | Stds (train vs test)    | Per-dim stds σ_train, σ_test       | MAE on z-scored stds                               | e^{-d} |
| Gen Q      | Gen→train vs test→train | Max Q distributions                | Wasserstein on z-scored samples                    | e^{-d} |
|            | Gen→test vs test→train  | Max Q distributions                | Wasserstein on z-scored samples                    | e^{-d} |
| Recon Q    | Train recon             | Recon train & test→train medians   | (1 − median(recon Q)) / (1 − median(test→train Q)) | e^{-d} |
|            | Test recon              | Recon test & test→train medians    | (1 − median(recon Q)) / (1 − median(test→train Q)) | e^{-d} |
| Clustering | Coord: gen+train        | Mixing ratio observed/expected     | max(0, 1 − ratio)                                  | e^{-d} |
|            | Coord: gen+test         |                                    | max(0, 1 − ratio)                                  | e^{-d} |
|            | Distmap: gen+train      |                                    | max(0, 1 − ratio)                                  | e^{-d} |
|            | Distmap: gen+test       |                                    | max(0, 1 − ratio)                                  | e^{-d} |
|            | Coord: recon train      |                                    | max(0, 1 − ratio)                                  | e^{-d} |
|            | Coord: recon test       |                                    | max(0, 1 − ratio)                                  | e^{-d} |
|            | Distmap: recon train    |                                    | max(0, 1 − ratio)                                  | e^{-d} |
|            | Distmap: recon test     |                                    | max(0, 1 − ratio)                                  | e^{-d} |


---

## 7. Implementation Notes

- **Generation data: variance = 1 only.** All generation-related scores (gen Rg/scaling/pairwise/avgmap, gen RMSD, gen Q, coord/distmap clustering gen) **must** be computed from data produced with **sample_variance = 1**. A different prior variance yields incomparable scores. Scoring reads the config’s `sample_variance` (or equivalent) for each block (plotting, rmsd_gen, q_gen, coord_clustering_gen, distmap_clustering_gen). Only when the list contains 1 (or 1.0) does scoring load that block’s gen data; and only from paths that explicitly indicate variance 1 (e.g. `gen_variance_1.0_data.npz`, or analysis gen run names like `default_var1.0`). If a block’s config does not include variance 1, those gen components are marked missing. Gen output paths always include the variance in the name (e.g. `gen/default_var1.0/`) so which data was used is unambiguous.
- **Modular:** Scoring is a separate step that reads existing pipeline outputs (NPZ, config). It does not run training or analysis. Pipeline components save only raw data (no precomputed scores); scoring loads that data and computes all component scores and the overall score.
- **Recon NPZ keys:** Analysis modules (rmsd.py, q_analysis.py) save recon data under `train_recon_rmsd`/`test_recon_rmsd` and `train_recon_q`/`test_recon_q`. Scoring accepts either those names or `recon_train_*`/`recon_test_*` so both conventions are found.
- **τ = 1 is fixed globally.** No per-metric scale config needed — z-normalization handles scale differences between metrics.
- **All 30 required:** The radar plot always shows all 30 components. Missing components are drawn with value 0 and their label in red. `overall_score` is computed only when all 30 components are present; otherwise it is set to NaN so composite scores are comparable across runs. Output lists which components are present vs missing.
- **Validation:** Run scoring on a set of known-good and known-bad models and confirm scores are well-separated (not all bunched near 0.9 or 0.1).
- **Train/test split** uses the pipeline seed (`split_seed`) and `training_split` consistently; sample sizes are controlled by existing config.

