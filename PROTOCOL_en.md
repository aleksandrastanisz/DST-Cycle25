# PROTOCOL: Dst forecasts and uncertainty-interval coverage during storms of solar cycle 25

**Version:** 1.2, 2026-11-27. Previous: 1.1 (frozen), 2026-11-24, [10.5281/zenodo.24511920](https://doi.org/10.5281/zenodo.24511920); 1.0, 2026-11-24, [10.5281/zenodo.24511114](https://doi.org/10.5281/zenodo.24511114). Differences are described in §13.

**Status:** frozen before the first training run and before any evaluation on test data. Any subsequent modification requires an entry in §13 (date, reason, impact).

**Scope:** historical data (NASA OMNI, Kyoto WDC). Conclusions do not apply to operational forecasting from real-time L1 data (§12).

---

## 1. Research question

> Do machine-learning forecasts of the Dst index add anything beyond persistence during intense storms of solar cycle 25, and do uncertainty-interval methods that aim to maintain coverage during storms keep that coverage when moved to a cycle they have not seen?

We do not ask which ML model is best, nor about lead times longer than 6 h. All models are built from scratch in this project; comparison with other authors' work happens only in the discussion, after the final evaluation (§12). Nor do we test whether standard conformal prediction has conditional coverage during storms: theory says it does not guarantee it, so it is reported only as a reference.

## 2. Definitions

**Dst** is the hourly ring-current index [nT]; more negative values mean a stronger storm. An hourly value is the mean over [t_start, t_start + 1 h) (§5.2).

**Point models** (one dataset, one evaluation procedure):

| ID | Model | Role |
|---|---|---|
| P | persistence: `Dst(t+h) = Dst(t)` | reference in H1 |
| E | O'Brien and McPherron empirical model (AK2) | second reference, exploratory; fitted for Dst > −150 nT, so it extrapolates in intense storms |
| X | XGBoost on the physical features defined in §7.1 | ML model, confirmatory |
| L | LSTM | ML model, confirmatory |

**Interval methods** (applied to every point model, nominal level 90%):

| ID | Method | Role |
|---|---|---|
| C-split | standard (split) conformal prediction | reference, descriptive |
| C-Mond | Mondrian: separate quantiles per class of Dst known at issue time | H2 |
| C-DtACI | adaptive conformal inference with dynamically tuned step size (Gibbs and Candès 2024) | H3 |
| C-ACI | adaptive conformal inference with a fixed step γ (Gibbs and Candès 2021) | sensitivity (S1) |

**Storm window** and **categories** are defined in §6. **Coverage** is the fraction of hours in which the observed Dst lies inside the interval. **Horizon** h ∈ {1 h, 6 h}; both carry equal weight.

## 3. What can be distinguished

Before interpreting results we accept that the following cannot be separated:

1. **Physical change versus label-version change.** Training and calibration use final Dst; the test period is mostly provisional (§5.1). The versions determine the baseline and Sq variation differently, so part of the distribution shift in the test period may come from the index version rather than from physics.
2. **Model value versus input data quality.** OMNI data are processed after the fact and time-shifted to the bow-shock nose. Results are an upper bound for operational forecasting. This also applies to Dst as a model input: at time t only real-time Dst is available operationally, with noise and baseline jumps, whereas the models receive the final or provisional version. This inflates the skill of the models and of persistence, most at h = 1 h.
3. **Storm-time coverage versus conditioning on the outcome.** Storm windows are defined from observed Dst (§6), so storm-time coverage is conditional on the outcome. Windows include every storm phase, which weakens this effect but does not remove it.
4. **Shift between cycles versus the difficulty of storms themselves.** We therefore also report storm-time coverage of C-Mond and C-DtACI on cycle 24, out of the calibration sample (§7.3, M9). A drop relative to cycle 24 measures shift, and this drop is what H2 tests; coverage below nominal on its own does not.
5. **Shift between cycles versus the weakness of the calibration cycle.** Cycle 24 was the weakest in a century (26 intense-storm windows, no superintense one), while cycle 25 is clearly stronger. An H2 result therefore concerns the move from a weak calibration cycle to a stronger test cycle and does not establish whether the same drop would occur between cycles of similar strength.

## 4. Hypotheses

Hypotheses are decided only by the rules in §10. Every result, including null results, is reported. Each hypothesis is tested for two ML models (X, L) and two horizons (1 h, 6 h), giving 4 tests per hypothesis.

- **H1 (ML advantage shrinks with storm intensity).** The skill score of an ML model relative to persistence is lower in intense-storm windows than in quiet hours.
- **H2 (conditional coverage drops on moving to a new cycle).** Coverage of C-Mond intervals in intense-storm windows is lower on cycle 25 than on cycle 24, measured out of the calibration sample (M9).
- **H3 (adaptation restores coverage).** C-DtACI intervals have higher coverage than C-Mond in intense-storm windows, and coverage of at least 85%.

Directional predictions, **exploratory** (no multiple-comparison correction, reported as observations):

- The ML advantage over persistence is larger at h = 6 h than at h = 1 h, because persistence loses value quickly with horizon.
- C-Mond loses coverage mainly at the start of the main phase, while Dst at issue time is still in a quiet class.
- C-DtACI pays for coverage with wider intervals mainly in the main phase and less in the recovery phase.
- Model E underestimates the minimum of intense storms less than the ML models, despite operating outside its fitted range.

## 5. Data

### 5.1 Sources and versions

| Variable | Source | Resolution | Notes |
|---|---|---|---|
| \|B\|, By and Bz in GSM, density, temperature, V, dynamic pressure, E field | [NASA OMNI2](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/), yearly files | 1 h | shifted to the bow-shock nose; fill values as in `omni2.text` |
| Dst | [Kyoto WDC](https://wdc.kugi.kyoto-u.ac.jp/dstdir/) via HAPI, dataset `hour_dst` | 1 h | version recorded for every hour |
| Dst in OMNI | OMNI2, column 41 | 1 h | used only to check hour alignment (§5.2) |

**Dst versions** come from the version code (`versionCode`) that the Kyoto HAPI server returns for every hour: 2x final, 1x provisional, 0x real-time. The `hour_dst` dataset serves the highest available version for each period. State at download (2026-11-24): final up to the end of 2020, provisional throughout 2021–2025. The script stops if train or calib contains any version other than final; version counts for each split are in `MANIFEST.json`.

### 5.2 Time convention

- `t_start` is the start of the hour whose mean the record describes.
- The value for the interval [t − 1 h, t) is known at time t. A forecast issued at time t uses only such values.
- The target of a forecast with horizon h is Dst for the interval [t + h − 1 h, t + h).
- **Notation:** `Dst(t)` denotes the value for the interval [t − 1 h, t), that is the record with `t_start = t − 1 h`. This applies to every formula with argument t (§2, §7, §7.1). In §6 and in the code, times denote the `t_start` of hourly records.
- Kyoto and OMNI may label hours differently. The offset is determined on 1995–2014 data by comparing Kyoto Dst with Dst in OMNI. We accept an offset with at least 99% agreement; if none reaches it, work stops until the cause is explained. At h = 1 h, a one-hour offset would leak the target into the input.

### 5.3 Splits (finite, frozen, on disk)

Data are downloaded once by `scripts/download_data.py`. The download date, Dst version counts per split, hour offset and SHA-256 checksums of all files are stored in `data/MANIFEST.json`.

| Split | Period (UTC) | Cycle (approx.) | Dst version | File | Use |
|---|---|---|---|---|---|
| train | 1995-01-01 – 2009-12-31 | 23 | final | `data/processed/train.parquet` | training and tuning (§7.2) |
| calib | 2010-01-01 – 2019-12-31 | 24 | final | `data/processed/calib.parquet` | conformal quantiles, warm-up of adaptive methods, M9 reference, power analysis; no model selection |
| test | 2020-01-01 – 2025-12-31 | 25 | final (2020), provisional (2021–2025) | `data/sealed/test.parquet` | single final evaluation |

Split boundaries are calendar dates, not exact cycle boundaries: train also contains the tail of cycle 22 (1995–1996), and cycle 24 began in late 2008. Each stage nevertheless covers mainly one cycle. Calibrating on all of cycle 24, rather than on 2015–2019, ensures storms from different phases of the cycle.

**The test split** is touched **only** in the final evaluation. It is not used for tuning, model selection, debugging or plots. Exception, computed from Dst alone, before any model is evaluated on the test split: the number of storm windows and quiet blocks in the test period (§6), which the power analysis also needs (§10.4).

## 6. Storms

**Definition** (fixed in advance, from the Dst series alone):

1. Storm minimum: an hourly record with `t_start` = s whose Dst is ≤ −50 nT and is the lowest among records with `t_start` in [s − 24 h, s + 24 h]. Ties go to the earliest record. At the edges of the data (1995-01-01 and 2025-12-31) the interval includes only the available records. Below, t_min denotes the `t_start` of the minimum.
2. Storm window: [t_min − 24 h, t_min + 48 h]. Overlapping windows are merged; the category of a merged window is set by its lowest minimum.
3. A window belongs to the split that contains its minimum (for a merged window: its lowest minimum) and is clipped to that split's boundaries.
4. A forecast hour belongs to a window if its **target hour** lies in the window.
5. **Quiet blocks:** target hours outside all windows are divided into consecutive 27-day calendar blocks (one solar rotation), starting at the beginning of the split. A block with fewer than 72 quiet hours is merged into the previous one, or into the next one if it is the first block of the split. Hours of a storm window assigned to the neighbouring split, cut off by clipping (item 3), are not quiet hours and belong to no category.
6. **Storm phases:** the main phase is the target hours from the start of the window up to and including the window's lowest minimum; the recovery phase is the target hours after that minimum. In windows merged from several storms, phases are defined relative to the lowest minimum.

**Categories** (−50 and −100 nT thresholds following Gonzalez et al. 1994; −250 nT threshold for superintense storms following Echer et al. 2008):

| Category | Minimum Dst [nT] | Role |
|---|---|---|
| quiet | blocks from item 5 | reference in H1 |
| moderate | −100 < min ≤ −50 | always reported |
| intense | min ≤ −100 | H1–H3 |
| superintense | min ≤ −250 (subset of intense) | descriptive only |

**Minimum number of storms.** If the test period contains fewer than **10** intense-storm windows, H1–H3 are reported descriptively, without a decision. The count is determined before any model is run and recorded in §15.

## 7. Models and training

- **Inputs to every model:** only values known at time t (§5.2). Current Dst is an input because its real-time counterpart is available operationally; the models nevertheless receive the final or provisional version (§3, item 2). This makes persistence a demanding reference.
- **P:** `Dst(t+h) = Dst(t)`.
- **E (AK2):** `dDst*/dt = Q − Dst*/τ`, `Q = −4.4·(VBs − 0.5)` nT/h for VBs > 0.5 mV/m and 0 otherwise, `τ = 2.4·exp(9.74 / (4.69 + VBs))` h, `Dst* = Dst − 7.26·√Pdyn + 11` (Dst in nT, Pdyn in nPa). VBs in mV/m: `V [km/s] · Bs [nT] · 10⁻³`, where Bs = −Bz for Bz < 0 and 0 otherwise. Integration with the explicit Euler scheme, 1 h step, from `Dst*(t)` over h steps; with a 1 h step the scheme is stable because τ > 2.4 h. Solar wind is held at its value at time t, since future values are unknown. Constants are not refitted.
- **X:** XGBoost (regression, squared loss) on the feature set in §7.1, one model per horizon. Missing feature values are handled natively. Every feature must pass the causality test (§14). Fixed: `subsample = 0.8`, `colsample_bytree = 0.8` (so that the seed affects the model), `tree_method = hist`, training on CPU; other parameters at the defaults of the frozen package version (§14).
- **L:** 24 h window of V, Bz and By in GSM, \|B\|, density, dynamic pressure, VBs and Dst; standardized with train statistics; one network per horizon; a linear layer on the last hidden state gives a single output; MSE loss, no dropout.
- **Every fitted transformation** (standardization, any imputers) is learned on train only.
- **Missing data:** inputs of all models are filled with the last known value, at most 3 h back and only from the past. After this step: P requires `Dst(t)`; E requires `Dst(t)` and V, Bz and Pdyn at time t; L requires a complete 24 h window of all channels; X handles the remaining gaps natively. A model without its required inputs issues no forecast for that hour. The main evaluation uses only the common set of hours in which **all** four models produce a forecast and the target is known. Because solar-wind data gaps may cluster in large storms, the fraction of excluded hours is reported for each category (M10) and S5 is run (§9).
- **Seeds:** 5 seeds (42–46) for X and L. The seed controls initialization and data order in L and subsampling in X. The point forecast is the mean of the 5 models; spread across seeds is reported separately.
- **Ensemble as the unit:** hypotheses concern the seed-averaged forecast, since that is the model one would deploy.

### 7.1 Features of model X (fixed in advance)

All features are computed only from values known at time t (§5.2); "lag k" means the value for the interval [t − k − 1 h, t − k). Running means and sums use complete windows only (after the gap filling in §7); a window with a gap gives a missing feature.

| Group | Features | Rationale |
|---|---|---|
| Ring-current state | Dst at lags 0, 1, 2, 3 and 6 h; `Dst(t) − Dst(t − 1 h)`; `Dst(t) − Dst(t − 3 h)` | current state and rate of change (main phase or recovery) |
| Solar wind | V, Bz and By in GSM, \|B\|, density, Pdyn at lags 0, 1, 2 and 3 h | direct driving |
| Coupling | VBs [mV/m]; Newell coupling function `V^(4/3)·B_T^(2/3)·\|sin(θ/2)\|^(8/3)`, where `B_T = √(By² + Bz²)`, `θ = atan2(By, Bz)` (IMF clock angle; absolute value because θ can be negative); `√Pdyn` | energy injection (Burton; O'Brien and McPherron) and the pressure correction in Dst* |
| Driving history | 3 h and 6 h running means of VBs, Bz, Pdyn and the Newell function; 6 h and 12 h sums of VBs | energy loaded before time t |

44 features in total: 7 + 24 + 3 + 10. The list is frozen with the protocol; no features are added or removed on the basis of results, including validation results. Feature importance (SHAP) is reported descriptively only.

### 7.2 Tuning (equal budget)

Tuning only within train, by cross-validation over blocks of years. Train is split into 5 consecutive 3-year blocks: 1995–1997, 1998–2000, 2001–2003, 2004–2006 and 2007–2009. In each fold one block is the validation block and the model is fitted on the others, with the 27 days adjacent to the validation block removed so that autocorrelation does not carry information between them. Criterion: mean validation MSE over the 5 folds; separate seed 100. Cross-validation replaces a single validation period because 2006–2009 is a deep solar minimum with only a few intense storms, and hyperparameters chosen on such a period would favour quiet conditions. Each ML model gets 12 configurations per horizon, tuned separately for h = 1 h and h = 6 h:

| Model | Grid |
|---|---|
| X | max_depth ∈ {4, 6, 8} × learning_rate ∈ {0.03, 0.1} × min_child_weight ∈ {1, 10}; number of trees by early stopping (patience 50, max 5,000) |
| L | hidden size ∈ {32, 64, 128} × lr ∈ {10⁻³, 3·10⁻³} × layers ∈ {1, 2}; Adam, batch 256, early stopping (patience 5, max 100 epochs) |

Early stopping in each fold uses that fold's validation block. The final model is trained on all of train (1995–2009) with the selected configuration; the number of trees or epochs is the median of the best iterations over the 5 folds of that configuration. If the optimum lies on the edge of the grid, the grid is not extended and this is reported with the results.

### 7.3 Intervals

Nonconformity score for all methods: `|Dst − forecast|`. Calibration residuals come from calib (models trained on train only), from 2010-01-28: the first 27 days of calib are skipped so that autocorrelation with the end of train does not shrink the residuals. This applies to all methods and to M9.

- **C-split:** the ⌈(n + 1)·(1 − α)⌉-th smallest of the n calibration residuals, for α = 0.1 (and for other levels in M13). The width is constant for a given model and horizon.
- **C-Mond:** as C-split, but with separate quantiles for classes of `Dst(t)` at issue time: `Dst(t) > −30`, `−50 < Dst(t) ≤ −30`, `−100 < Dst(t) ≤ −50`, `Dst(t) ≤ −100` nT. The class depends on the issue time, not on the outcome. A class with fewer than 200 hours in calib is merged with its neighbour closer to zero. Merging is decided once, on all of calib, and applied unchanged in the test period and in M9; class sizes are reported.
- **C-DtACI:** Gibbs and Candès (2024), deterministic variant (Algorithm 2: α(t) is the weighted average of the experts' levels rather than a single sampled one). k = 8 ACI experts with steps γ ∈ {0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128}, as in the paper. Weights are updated with the pinball loss `ℓ(β, θ) = α(β − θ) − min{0, β − θ}`, where β(t) is the largest level at which the target lies in the interval and α = 0.1. Mixing `σ = 1/(2I)`, weight step `η = √(3/I) · √((ln(k·I) + 2) / ((1 − α)²α²))` (formulas from the paper, with its approximation of the loss). Interval I = 72 h, the storm-window length, instead of 500 in the paper; this gives σ ≈ 0.0069 and η ≈ 6.6. Quantile of order 1 − α(t) of the calibration residuals.
- **C-ACI (S1):** a single expert, `α(t+1) = α(t) + γ·(0.1 − err(t))`, γ ∈ {0.005, 0.05}.
- **Rules shared by the adaptive methods:**
  - **Feedback delay:** a forecast issued at time t yields `err` only at t + h. The update at time t uses only forecasts whose target is already known.
  - Start: α = 0.1 (and equal expert weights) on 2010-01-01; runs through all of calib and on through test without reset.
  - Hours without a forecast (§7, missing data) are skipped: they give neither an interval nor an update.
  - When α ≤ 0 the interval is infinite and counted as covered; when α ≥ 1 it is empty and counted as not covered. The number of such hours is reported, and width is reported as a median.
- **Cycle-24 reference (M9):** C-Mond and C-DtACI computed on the two halves of calib by cross-fitting: quantiles from 2010–2014 applied to 2015–2019 and vice versa. Residuals are split by hour, and a storm window is assigned to the half that contains its lowest minimum. For C-DtACI the residual distribution comes from one half, and the online run through the other half starts from the initial state (α = 0.1, equal weights). This gives intense-storm coverage out of the calibration sample but within the calibration cycle.

## 8. Metrics

All computed on the common set of hours (§7), separately for each model, horizon and category from §6.

**The common set of hours** is determined separately for each horizon and is the same for all interval methods. Ablations (§9) are evaluated on the E-main set; hours in which the ablated model has no forecast are additionally excluded and their number is reported.

**A unit enters the analysis** if at least half of its target hours are in the common set; other units are excluded and counted in M10.

**Aggregation.** The primary statistic is a mean over **units** (storm windows; quiet blocks in the quiet category), each unit with equal weight. The hour-weighted version is reported as secondary. For the skill score we aggregate errors, not ratios:

`SS_cat = 1 − mean_j MSE_model,j / mean_j MSE_P,j`, where j runs over the units of the category.

**Primary:**

| ID | Definition |
|---|---|
| M1 | `SS_cat` as above |
| M2 | coverage: mean over units of the fraction of hours in which the target lies in the interval |

**Secondary:**

| ID | Definition |
|---|---|
| M1b | difference `mean_j MSE_P,j − mean_j MSE_model,j` [nT²]; complements M1, because persistence MSE is small in quiet hours and the ratio can be unstable |
| M3 | RMSE and MAE [nT] |
| M4 | median interval width [nT] |
| M5 | interval score for α = 0.1: `(u − l) + (2/α)·(l − y)·1[y < l] + (2/α)·(y − u)·1[y > u]` [nT] |
| M6 | SS relative to model E: `1 − mean_j MSE_model,j / mean_j MSE_E,j` |
| M7 | hidden-persistence check: the shift k ∈ {0, …, h + 3} h that maximizes the correlation between forecast and observation within a storm window; median over windows and fraction of windows with k ≥ h (cf. Laperre et al. 2020); windows with a constant forecast are skipped |
| M8 | per window: forecast minimum minus observed minimum [nT] and timing error of the minimum [h]; ties go to the earliest hour |
| M9 | intense-storm coverage of C-Mond and C-DtACI on cycle 24, cross-fitted (§7.3); drop `δ = M9 − M2(test)`, which for C-Mond is the H2 statistic |
| M10 | fraction of hours excluded from evaluation because of missing data and number of units excluded by the half-of-hours rule, per category |
| M11 | dose–response curve: M1 and M2 computed for each window separately as a function of the window's minimum Dst; scatter plot and means in 50 nT bins of the minimum |
| M12 | point-forecast shift: M3(test) / M3(calib) per category; calib is out of the training sample for the models |
| M13 | coverage curve: M2 at nominal levels 50, 60, 70, 80, 90 and 95% (C-split, C-Mond, C-DtACI), separately for quiet hours and intense storms |

M1–M5 are also reported separately for storm phases (§6, item 6).

## 9. Experiments

| ID | Description | Role |
|---|---|---|
| R0 | pipeline correctness checks on out-of-fold predictions from cross-validation (train), §9.1 | prerequisite; no test data |
| P0 | power analysis on cycle 24 (§10.4) | informational; no test data |
| E-main | P, E, X, L × h ∈ {1, 6} × {C-split, C-Mond, C-DtACI} on test | confirmatory (H1, H2, H3) |
| S1 | C-ACI with γ ∈ {0.005, 0.05} | sensitivity to the adaptation scheme |
| S2 | hour-weighted statistics instead of unit-weighted | sensitivity to aggregation |
| S3 | storm windows [t_min − 12 h, t_min + 24 h] instead of §6 | sensitivity to the window definition |
| S4 | M6 and an H1 analogue relative to model E | whether conclusions depend on the choice of reference |
| S5 | P and X on all hours for which they themselves produce a forecast (XGBoost handles missing values) | sensitivity to missing-data exclusions |
| S6 | leave-one-storm-out: every confirmatory statistic recomputed without each intense-storm window of the test period in turn | stability of decisions |
| S7 | the H2 statistic with test coverage computed from C-Mond quantiles of each calib half separately (2010–2014 and 2015–2019), that is with the same calibration-sample size as M9; both values and their mean are reported | symmetry of the H2 comparison |
| X1 | storm windows whose lowest minimum falls on 10–11 May or 10–11 October 2024: M2, M4 and M8 | exploratory, descriptive only |
| X2 | M1–M3 for all models and interval methods, separately for 2020 (final Dst) and 2021–2025 (provisional Dst); categories without windows in a period are omitted | exploratory; effect of the label version (§3, item 1) |
| X3 | H1–H3 counterparts for the superintense category | descriptive only |
| X4 | one repeat of training L (seed 42, both horizons) with the same configuration | scale of run-to-run noise from training nondeterminism (multithreaded CPU); descriptive only |
| A1 | X and L without any Dst inputs (X without the "ring-current state" group, L without the Dst channel) | ablation: whether the advantage over persistence comes from the solar wind or from extending current Dst |
| A2 | X without the "coupling" group | ablation: contribution of the coupling functions |
| A3 | X without the "driving history" group | ablation: contribution of energy loaded before time t |
| A4 | X on the "ring-current state" group only | ablation: purely autoregressive model, complement to A1 |

**Ablations A1–A4** use seeds 42–46, the configurations selected when tuning the full model (no re-tuning), the same data splits and all three interval methods. They therefore measure sensitivity at fixed hyperparameters. For each ablation we report M1, M2 and M7 per category and the difference from its E-main counterpart (same model, horizon and seeds), with a 95% bootstrap interval (§10.1).

### 9.1 Correctness checks (R0)

Run on out-of-fold predictions from cross-validation of the selected configuration, over the whole train period, before any contact with calib or test. Failure of any check means a bug to find, not a result; the bug and its fix are recorded in §13.

1. **Positive control:** X and L must have lower MSE than P at both horizons. A model that does not beat persistence on data from the same cycle is most likely wired incorrectly.
2. **Target-leakage control:** X trained with the target added as a feature must reach MAE close to zero (< 1 nT). This confirms the pipeline can detect leakage when it exists.
3. **Negative control:** X trained on a target randomly shuffled within train must not beat a constant forecast (the train mean) by more than 1% in MSE.
4. **Model E:** (a) unit test: under constant driving VBs = 5 mV/m and Pdyn = 2 nPa, `Dst*` converges to the steady state `Q·τ` within 1%; this checks units (VBs in mV/m, Pdyn in nPa) and the integration. (b) Correlation between forecast and observation > 0.8 at h = 1 h; this only checks that the model is wired in, because at h = 1 h the correlation comes mainly from `Dst(t)`.

**Analysis of S1–S5, S7, A1–A4 and X1–X4** is descriptive, without hypothesis tests: point estimate and 95% bootstrap interval (§10).

**Run order** (most important for interpretation first): tuning, R0, P0, E-main, S6, S7, A1, A4, S5, S2, S4, A2, A3, S1, S3, X4, X1, X3, X2.

## 10. Statistics and decision rules

### 10.1 Units and bootstrap

**The independent unit** is a storm window or a quiet block (§6), not an hour, because hours within one storm are strongly correlated.

**Bootstrap.** In each replicate, storm windows are resampled with replacement separately within each category, and quiet blocks are resampled likewise. One set of draws is shared by all models, horizons and interval methods, because comparisons are paired. 10,000 replicates, seed 2026. Percentile 95% intervals. The one-sided p-value is `(b + 1)/(B + 1)`, where b is the number of replicates in which the statistic lies on or beyond the null boundary (0 for H1, H2 and H3) and B = 10,000; p is therefore never zero. The calib intense-storm windows needed for M9 in H2 are resampled independently of the test windows (separate random stream, seed 2028), jointly for all models and horizons.

**Scope of inference.** The bootstrap covers the variability of the test set (storms and blocks). Inference concerns the trained ensemble, that is the mean of 5 seeds; it does not cover training variability, which is therefore reported descriptively (spread across seeds, X4).

**Correction.** Holm's method separately within each family: 4 tests per hypothesis (X, L × 1 h, 6 h), α = 0.05. Type I error is controlled separately within each of the three families (H1, H2, H3); no correction is applied across families.

**Practical thresholds** (0.05 for the skill score, 5 percentage points of coverage in H2, 3 points for the paired difference in H3) are fixed in advance as the smallest differences we consider practically relevant to a forecast user. They are not derived from the data; P0 shows which effects are detectable at these thresholds.

### 10.2 Rules

**H1.** For model v and horizon h: `Δ = M1(quiet) − M1(intense)`. H1 is supported for (v, h) when both:

- `Δ ≥ 0.05` (practical threshold),
- Holm-adjusted p < 0.05.

**H2.** For C-Mond on model v and horizon h: `δ = M9(intense) − M2(test; intense)`, the drop in intense-storm coverage from cycle 24 to cycle 25. H2 is supported when both:

- `δ ≥ 0.05` (a drop of at least 5 percentage points),
- Holm-adjusted p < 0.05 for `δ > 0`.

The absolute coverage M2(test; intense) is reported alongside. The H2 conclusion concerns the move from the weak cycle 24 to the stronger cycle 25 (§3, item 5).

**H3.** `D = M2(intense; C-DtACI) − M2(intense; C-Mond)`. H3 is supported when all of:

- `D ≥ 0.03`,
- Holm-adjusted p < 0.05 for `D > 0`,
- point estimate `M2(intense; C-DtACI) ≥ 0.85`.

The ratio of median widths of C-DtACI to C-Mond is also reported; it does not enter the rule.

### 10.3 Robustness

- **Few clusters.** A bootstrap over a dozen or so clusters can be anti-conservative. For H3, where the comparison is paired within each window, a sign-flip permutation test on the per-window differences `D_j` is also reported (100,000 random sign flips, seed 2027; p computed as in §10.1). The rule in §10.2 decides; any disagreement with the permutation test is flagged in the report.
- **Influence of a single storm (S6).** If removing one window changes the decision for any hypothesis, the decision remains formally valid but is reported as **fragile**, naming that storm.
- **Missing data.** If M10 in the intense category exceeds 10%, H1–H3 results are reported with a caveat and alongside S5.

### 10.4 Power analysis (P0, informational)

P0 is run after the protocol is frozen, before the test evaluation. It is informational only: it does not change thresholds, rules or which hypotheses are decided. It serves to interpret a possible null result.

1. On calib, with models trained on train and with C-Mond and C-DtACI intervals cross-fitted as in M9, compute the statistics in §10.2.
2. Estimate their standard error SE by bootstrap (10,000 replicates, seed 2029), resampling with replacement as many intense-storm windows and quiet blocks as the test period contains (§6; the number of quiet blocks is counted at P0, from Dst only). For H2, `SE = √(SE_M9² + SE_test²)`: the first term from resampling the 26 calib windows, the second from resampling as many windows as the test period contains, both on calib data.
3. A hypothesis is supported when the estimated distance from the H0 boundary exceeds both the practical threshold d and z·SE, where z = z_{1−0.0125} ≈ 2.24 (the strictest Holm level with 4 tests). Under the normal approximation the smallest effect detectable with 80% power is `MDE80 = max(d, 2.24·SE) + 0.84·SE`, with d = 0.05 for H1 and H2 and 0.03 for H3.
4. MDE80 is reported for each hypothesis, model and horizon. The C-DtACI coverage condition ≥ 0.85 in H3 is ignored in this approximation, and this is stated in the report.

**Additionally (always):** M1–M13 for all models, horizons and categories, with 95% bootstrap intervals; spread of M1 and M3 across seeds; C-split as a reference.

**A null result is a result.** If a hypothesis is not supported, the report says so and shows the bootstrap distributions. The endpoint, thresholds, categories and storm definition must not be changed after seeing the results.

## 11. Threats to validity

| Threat | Mitigation |
|---|---|
| Different Dst versions in training and test | Stated in §3; X2 compares the year with final Dst against the years with provisional Dst; re-evaluation once final Dst is published (out of scope) |
| OMNI data processed after the fact | Scope of conclusions (§12) |
| Leakage through an hour offset | Alignment check on 1995–2014 (§5.2) and feature causality test (§14) |
| Non-causal features or transformations fitted on all data | Causality test for every feature (§14); transformations fitted on train only; R0 checks (§9.1) |
| Features chosen to fit the result | Feature list frozen with the protocol (§7.1); no feature selection |
| Hypotheses formulated after reading papers with cycle-25 results (e.g. Yu et al. 2026: pooled coverage below nominal) | Disclosed in the paper; blinding covers evaluation of our models, not knowledge of the literature |
| The author knows how the 2024 storms unfolded | Blinding covers model evaluation, not knowledge of events; storm definitions and rules fixed before evaluation |
| Storm windows defined from observed Dst | §3; S3 tests sensitivity to the window definition |
| Few intense storms in the test period; bootstrap with few clusters | 10-storm threshold (§6); power analysis P0; permutation test for H3; superintense category descriptive only |
| A single storm (e.g. May 2024) dominates the statistics | Equal weighting of windows (§8); S6 and flagging of fragile decisions |
| Missing data clustered in storms | M10, 10% threshold and S5 |
| Unstable skill score in quiet hours | M1b as a complement; aggregating errors rather than ratios |
| Model E outside its fitted range in intense storms | Exploratory only; stated in §2 |
| Coverage below nominal may reflect storm difficulty rather than cycle shift | H2 tests the drop relative to cycle 24 (M9), not the absolute level (§3, item 4) |
| Weak calibration cycle (24) versus a stronger test cycle (25) | §3, item 5; narrowed wording of the H2 conclusion (§12) |
| Dst input in a better version than is available operationally | §3, item 2; skill at h = 1 h as an upper bound (§12) |
| Hyperparameters chosen for quiet conditions | Cross-validation over blocks of years instead of a solar-minimum period (§7.2) |
| Asymmetry in H2: M9 uses quantiles from half of calib (5 years), the test from all of calib (10 years). A smaller sample gives more conservative quantiles and more spread, so M9 may be slightly inflated, which favours H2; the storm-level bootstrap does not cover the uncertainty of the calibration residuals | Stated as a limitation; S7 computes test coverage with the same calibration-sample size as M9 |
| Asymmetry in H3: C-DtACI without classes, C-Mond with classes | Stated as a limitation; a combined variant is out of scope |
| Hourly residuals are not exchangeable (autocorrelation) | The C-split and C-Mond guarantees are not exact even on calib; coverage on calib is reported as a check |
| Hyperparameters tuned on cycle 23 only | Limitation; equal tuning budget for X and L (§7.2) |
| Storms are rare and training does not up-weight them | Deliberate: we evaluate the common practice of MSE training; the limitation is discussed |
| Choice of step size in the adaptive method | C-DtACI tunes its own step; γ grid and interval I fixed in advance; C-ACI with fixed γ in S1 |
| Data revisions after download | Single download, SHA-256 checksums and date in `MANIFEST.json` |
| The ML advantage comes from extending current Dst rather than from the physics of driving | A1 and A4; M7 check |
| A bug in the statistics implementation | The §10 code is validated on synthetic data with a known effect before the final evaluation (§14) |
| Nondeterminism of LSTM training (multithreaded CPU) | X4 as a noise measure; forecast as the mean of 5 seeds |

## 12. Scope of conclusions

Conclusions apply to: Dst forecasts at horizons of 1 h and 6 h, the OMNI and Kyoto data in §5, models P, E, X and L built in this project, trained on 1995–2009 and evaluated on 2020–2025, the interval methods in §2, and the storm definitions in §6. They **do not apply** to operational forecasting from real-time L1 data, other indices (SYM-H, Kp), longer horizons, or models not listed in §2. Conclusions about superintense storms (≤ −250 nT) are descriptive. The H2 conclusion concerns the move from the weak cycle 24 to the stronger cycle 25 (§3, item 5). Skill at h = 1 h is an upper bound also because the Dst input is not the real-time version (§3, item 2). The models never see cycle 24 data: they are trained on 1995–2009 and evaluated on 2020–2025 without retraining, so conclusions from H1 concern models with a gap of more than a decade in their training data; an operational model would be retrained on more recent data.

**Comparison with the literature** (e.g. Sandu and Beșliu-Ionescu 2026, Yu et al. 2026) is descriptive only and takes place in the discussion, after the final evaluation. Differences in test periods, Dst versions and metrics mean the numbers are not directly comparable; it does not affect any decision.

## 13. Deviations log (changes after freezing)

| Date | What changed | Reason | Impact on analysis |
|---|---|---|---|
| 2026-11-24 | Disclosure (before freezing): while checking the format of the downloaded Kyoto files, the first and last rows of the 2020, 2021, 2024 and 2025 files (8 hourly Dst values in total) and the counts of version codes in those years were displayed | Transparency | Negligible; no model was evaluated, and storm definitions, rules and thresholds do not depend on these values |
| 2026-11-24 | Zenodo DOI of the archived protocol added to §14 | Public, immutable timestamp of the frozen protocol | None |
| 2026-11-24 | New row in §11 (asymmetric calibration-sample size in H2) and experiment S7 in §9 | Post-freeze review: M9 and the test use quantiles from samples of different size | Exploratory; H2 rule unchanged; added before any training |
| 2026-11-24 | Sentence in §12 on the models not being retrained on cycle 24 data | Post-freeze review: scope of H1 conclusions | None; clarifies the scope of conclusions |
| 2026-11-24 | Release of version 1.1 (tag `protocol-v1.1`, new archive version on Zenodo) | Collecting the post-freeze review changes into one archived version, before the first training run | None |
| 2026-11-24 | Zenodo DOI of version 1.1 added to §14 | Public, immutable timestamp of version 1.1 | None |
| 2026-11-26 | Execution of step 1 in §15: smoke test (`scripts/smoke_test.py`) on an Intel Core i5-1135G7 CPU (4 cores, 8 threads), xgboost 3.4.1, torch 2.15.0 (CPU); one cross-validation fold (block 2001–2003), h = 1 h. XGBoost 0.1–0.2 min per run (about 20 min for 170 runs); LSTM 0.1–0.3 min per epoch (about 8–38 h for 142 runs at 20–100 epochs). Exact package versions saved in `requirements-lock.txt` | Plan in §15 | None; train data only, no validation errors printed |
| 2026-11-26 | LSTM trained on CPU instead of GPU (changes in §9 for X4, in §11, §14 and §15) | The laptop has no NVIDIA GPU; the estimated time is acceptable, and one machine and one environment for all models limit the risk of differences between machines | Multithreaded CPU training is not necessarily bitwise reproducible either; X4 still measures run-to-run differences. Hypotheses and rules unchanged |
| 2026-11-26 | Clarification of tuning details (§7.2) left open by the protocol; all are implemented in commit `0cd315f`, from which tuning was started: (1) validation MSE of each model computed on the hours in which that model issues a forecast and the target is known (only configurations of the same model are compared); (2) L trained on a target standardized with statistics of the training part, forecast converted back to nT, early stopping by MSE in nT²; (3) number of trees of X is the best iteration + 1, forecasts use only those trees; (4) ties in mean MSE broken by grid order; (5) "edge of the grid" reported for parameters with at least 3 values (depth of X, hidden size of L) | These details are missing from §7.2; fixed in code before any validation error was seen; entry added while X was being tuned | No effect on the hypotheses; (2) does not change the minimizer of the loss |
| 2026-11-27 | Execution of steps in §15: tuning (§7.2) and R0 checks (§9.1). X tuned from commit `0cd315f` (about 10 min), L from commit `93e5e9e` (code identical to `0cd315f`; the commits in between add only the §13 entry and the X tuning results; about 3.5 h; best epochs 1–24, so much shorter than the smoke-test estimate). Selected configurations: X h = 1 h depth 6, lr 0.03, min_child_weight 10, 597 trees; X h = 6 h depth 4, lr 0.03, min_child_weight 1, 180 trees; L h = 1 h hidden size 128, lr 10⁻³, 1 layer, 9 epochs; L h = 6 h hidden size 128, lr 10⁻³, 2 layers, 3 epochs. Optimum on the edge of the grid: X h = 6 h (depth 4, minimum) and L at both horizons (hidden size 128, maximum); the grids were not extended. R0 from commit `d66086f`: all checks passed. Check 1 computed on the common set of hours (as in the main evaluation); checks 2 and 3 for both horizons, with the selected configuration of X and the folds and early stopping of tuning; the target shuffle in check 3 uses seed 100. Results in `results/tuning/` and `results/r0.json` | Plan in §15 | None; train data only |
| 2026-11-27 | Execution of step 3 in §15: C-Mond class sizes on calib (`scripts/mondrian_classes.py`, commit `66130f9`), from 2010-01-28, on the common set of hours (all four models forecast and the target is known). h = 1 h: 76,176 / 6,915 / 2,280 / 216 hours (Dst(t) > −30 / (−50, −30] / (−100, −50] / ≤ −100); h = 6 h: 76,171 / 6,915 / 2,280 / 216. No class has fewer than 200 hours, so no classes were merged; this decision holds in the test period and in M9, although in each half of calib the ≤ −100 class has fewer hours (§7.3). Merging was implemented from the most negative class, merging again when a merged class still has fewer than 200 hours (irrelevant here). Calib forecasts from the final models (`scripts/train_final.py`, commit `0cf75a5`; `scripts/forecast_calib.py`, commit `7ced628`); results in `results/final_models.json`, `results/forecasts/calib_summary.json` and `results/mondrian_classes.json` | Plan in §15 | None; the ≤ −100 class has only 216 hours, which must be taken into account when interpreting H2 |
| 2026-11-27 | Clarification of implementation details left open by the protocol, before P0 was run (code in commits `ae545ba`, `553ea56` and `f9d9ab0`). **Units (§6):** a quiet block with fewer than 72 hours is merged into the previous block or, when no earlier block remains (the first block or a chain of small blocks at the start), into the next one; merging cascades. **Intervals (§7.3):** the quantile of order 1 − β is the ⌈(n + 1)(1 − β)⌉-th smallest residual; an index above n gives an infinite interval and β ≥ 1 an empty one (the same in C-split, C-Mond and the adaptive methods). Calibration residuals and online runs use only hours of the common set (§7), residuals from issue time 2010-01-28. With delay h, the update from row i is applied at row i + h of the hourly grid and uses the levels of the interval issued at row i. DtACI weights are normalized after every step (this does not change α(t)). The paper by Gibbs and Candès gives no numerical example, so the agreement test in §14 compares the implementation with an independent, literal transcription of Algorithm 2 (agreement to 10⁻¹⁰). **Halves of calib (M9):** rows are split by target hour (2010–2014 / 2015–2019), except rows of a storm window that crosses the boundary between the halves: these go to the half containing the window's lowest minimum, so that a window's hours do not calibrate its own intervals (P0 reports the number of such windows); a half's calibration residuals are its rows in the common set, in the 2010–2014 half from 2010-01-28; the online run through the other half goes over its rows in time order. **Bootstrap (§10.1):** one count matrix of draws per category, shared by models, horizons and methods; units excluded by the half-of-hours rule drop out of a replicate; a replicate with no included unit is skipped for p and the interval. On synthetic data without an effect (24 units, 400 data sets) a one-sided test at 0.05 rejected in 6.75% of cases, which confirms the risk described in §10.3. **P0 (§10.4):** one generator with seed 2029, draws in this order: 82 quiet blocks out of 136, 24 intense windows out of 26 (shared by H1, the test term of H2 and H3), 26 intense windows out of 26 (the M9 term of H2); coverage from intervals cross-fitted as in M9; H1 from the calib forecasts | These details are missing from §6, §7.3 and §10; fixed before P0 was run and before any interval was computed on data | No effect on the hypotheses or rules |
| 2026-11-27 | Model, interval and bootstrap parameters are constants in the `dstc25/` modules (e.g. `xgb_model.GRID`, `lstm_model.GRID`, `intervals.DTACI_GAMMAS`, `bootstrap.SEED_TEST`) instead of `configs/*.yaml` files (change in §14) | A single source of values in tested code, with no risk of a mismatch between a configuration file and the code; the commit determines every value | None; reproducibility is ensured by the commit hash |
| 2026-11-27 | Execution of P0 in §15 (`scripts/p0.py`, commit `22a6bbe`; result in `results/p0.json`). No storm window crosses the boundary between the halves of calib; all calib units satisfy the half-of-hours rule (136/136 quiet blocks, 26/26 intense windows). Calib values and MDE80 (X h = 1, L h = 1, X h = 6, L h = 6): H1 Δ = 0.107 / −0.160 / −0.168 / −0.151, MDE80 = 0.561 / 0.100 / 0.107 / 0.104; H2 M9 (C-Mond) = 0.834 / 0.838 / 0.764 / 0.753, MDE80 = 0.067 / 0.065 / 0.066 / 0.065; H3 M9 (C-DtACI) = 0.874 / 0.869 / 0.817 / 0.809, D = 0.040 / 0.031 / 0.052 / 0.056, MDE80 = 0.047 / 0.041 / 0.053 / 0.047. The large standard error of H1 for X at h = 1 h (0.182) comes from the ratio of mean MSEs in replicates with few intense windows | Plan in §15 | None; P0 is informational only (§10.4) and changes no hypothesis, threshold or test |
| 2026-11-27 | Disclosure of test-data use before the final evaluation: `scripts/count_units.py` and `scripts/p0.py` read only the `t_start` and `Dst` columns of `data/sealed/test.parquet`, as `scripts/count_storms.py` did before. They serve to define storm windows from Dst alone (a minimum near the calib/test boundary is searched within ±24 h, §6 item 1) and to count the units in the test period (§5.3, §10.4). Only unit counts are displayed and saved; no model was evaluated on test data | Exception provided for in §5.3 | None |
| 2026-11-27 | Release of version 1.2 (tag `protocol-v1.2`, new Zenodo archive version): text changes logged in §13 since 2026-11-26 (§9 for X4, §11, §14, §15) and the results of the §15 steps before any test evaluation (tuning, R0, final models in `models/`, C-Mond class sizes, P0) | Archiving before any evaluation on test data, as required by §14 | No effect on the hypotheses; the final evaluation requires a commit descending from `protocol-v1.2` |
| 2026-11-27 | Zenodo DOI of version 1.2 added to §14 | Public, immutable timestamp of version 1.2 and of the results obtained before test evaluation | None |
| 2026-11-28 | Clarification of E-main details left open by the protocol, before the test split is opened (`scripts/e_main.py`, `dstc25/decisions.py`, `dstc25/main_intervals.py`, `dstc25/guard.py`). **Draws:** a generator with seed 2026 draws, in this order, the quiet blocks, moderate windows and intense windows of the test period; a separate generator with seed 2028 draws the 26 calib intense windows for M9; in both cases units are taken in the order of their identifiers. **H3 permutation test:** b is the number of random sign flips for which the mean of D_j is not smaller than the observed one. **S6:** test intense windows are removed one at a time (the only storm category in H1–H3); for each removal the p-values, Holm's correction and the decisions are computed again. **Forecasts:** calib and test are computed together with the archived models; the calib part must equal `results/forecasts/calib.parquet` (tolerance 10⁻³ nT), otherwise the script stops. The last h calib hours, whose targets lie in the test period, are not calibration residuals but do give feedback to the adaptive methods and are evaluated with the test period, because the target hour decides membership (§6 item 4). **Test openings:** each one is recorded in `results/test_openings.json` before the test file is read; a rerun requires committing this log (clean repository) | These details are missing from §10; fixed before the test split is opened | No effect on the hypotheses or rules |
| 2026-11-28 | Execution of E-main (`scripts/e_main.py`, commit `3ae177b`; one opening of the test split, recorded in `results/test_openings.json`; results in `results/e_main/`). Test units: 82 quiet blocks, 81 moderate and 24 intense windows (2 of them superintense); the half-of-hours rule excluded 1 intense and 1 moderate window (M10 in the intense category about 3%, no caveat under §10.3). H1 supported only for X at h = 1 h (Δ = 0.387), a fragile decision (S6). H2 supported for X at h = 1 h (fragile), X at h = 6 h and L at h = 6 h; for L at h = 1 h δ = 0.047 is below the threshold. H3 supported for X at h = 1 h, L at h = 1 h and X at h = 6 h (fragile, M2 of C-DtACI = 0.852); for L at h = 6 h the condition M2 ≥ 0.85 is not met (0.845). The permutation test agrees with the bootstrap in all four H3 comparisons | Plan in §15 | Results deciding H1–H3 |
| 2026-11-28 | Clarification of the secondary metrics before they were computed (`dstc25/secondary.py`, `scripts/secondary_metrics.py`): M7 – a tie in correlation goes to the smaller shift, a window needs at least 3 hour pairs; M8 – the minimum and its time are taken over the window's common-set hours; M11 – bins (lo, lo + 50] nT, matching the category edges, M1 within a bin aggregated as in §8; M12 – hour-pooled RMSE on the common set; phases – per-unit means over the hours of the phase, then the mean over included units. In addition, for the interpretation of H3: the share of hours with an infinite interval in intense windows and coverage on hours with a finite interval (descriptive, no effect on the rule in §10.2). The script reads the test rows saved by E-main (checked by SHA-256) and is recorded in the openings log | These details are missing from §8 | No effect on the hypotheses |

Changes made before freezing are visible in the Git history. Any use of test data before the final evaluation, including in code tests, is disclosed in this table, even if no model was evaluated on it.

## 14. Reproducibility

- Data: `scripts/download_data.py`, run once; `data/MANIFEST.json` with download date, URLs, Dst version counts per split, hour offset, package versions and SHA-256 checksums.
- All model, interval and bootstrap parameters are constants in the `dstc25/` modules (§13, 2026-11-27). The report includes the commit hash.
- Exact versions of all packages (`pip freeze`) are saved in the repository (`requirements-lock.txt`) before tuning and are not changed until the evaluation is complete; a version change is an entry in §13.
- Automated tests: OMNI parser and fill values, hour alignment, split disjointness, storm-window definition on synthetic data, causality of every feature (changing data after time t does not change the feature at time t), feedback delay in C-ACI and C-DtACI, steady state of model E under constant driving, agreement of the C-DtACI implementation with an example from Gibbs and Candès.
- The final evaluation script refuses to run if the SHA-256 of `data/sealed/test.parquet` does not match `MANIFEST.json` or the current commit is not a descendant of the tag of the latest protocol version (currently `protocol-v1.2`).
- The implementation of the bootstrap, Holm correction and rules in §10 is validated on synthetic data with a known effect (with and without an effect) before the final evaluation.
- Every run records the commit and an uncommitted-changes flag at start. E-main and ablation runs are launched only from a clean repository.
- XGBoost on CPU is reproducible with the same seed. LSTM is trained on CPU (§13, 2026-11-26); multithreaded training is not necessarily bitwise reproducible (X4 measures the scale of differences).
- `scripts/make_tables.py` regenerates every table and figure from saved forecasts, without manual steps.
- The frozen protocol with the code is archived on Zenodo at freezing; the results of the steps planned after freezing (§15), including R0 and P0, go into a later version of the archive, before any evaluation on the test split. Version `protocol-v1.0`: [10.5281/zenodo.24511114](https://doi.org/10.5281/zenodo.24511114); version `protocol-v1.1`: [10.5281/zenodo.24511920](https://doi.org/10.5281/zenodo.24511920); version `protocol-v1.2`: [10.5281/zenodo.24602843](https://doi.org/10.5281/zenodo.24602843); all versions: [10.5281/zenodo.24511113](https://doi.org/10.5281/zenodo.24511113).

## 15. Decisions prior to freezing (resolved)

**Resolved:**

- **Horizons:** 1 h and 6 h with equal weight, with Holm correction over 4 tests per family.
- **Split:** mainly by cycles 23 / 24 / 25 (§5.3); the test period covers 2020–2025, entirely final or provisional (change from version 0.4, where the test period ended in June 2025 according to the Kyoto page; the HAPI version codes showed that all of 2025 is already provisional).
- **Nominal level:** 90%, because it yields more out-of-interval cases during storms than 95%, and therefore more power.
- **Unit of analysis:** storm window or quiet block; equal weighting of units.
- **Confirmatory methods:** C-Mond (H2) and C-DtACI (H3). C-split is not tested, because it does not promise conditional coverage (change from version 0.1).
- **Models:** all built from scratch in the project; model X features frozen in §7.1 (change from version 0.2, in which model X was based on the Sandu and Beșliu-Ionescu code).
- **Hour alignment:** the Kyoto-to-OMNI offset is 0 h, with 100% agreement on 1995–2014; other offsets give 7–15% agreement (`MANIFEST.json`, download of 2026-11-24).
- **OMNI columns:** column numbers and fill values in `OMNI_COLS` match `omni2.text`.
- **Model E constants:** b = 7.26 nT·nPa^−1/2 and c = 11 nT, following O'Brien and McPherron (2000).

- **Tuning:** separately for each horizon, grids in §7.2, cross-validation over 5 blocks of 3 years.
- **H2:** tests the drop in C-Mond coverage relative to cycle 24 (M9), with independent bootstrap of windows from both cycles.
- **C-DtACI:** deterministic variant, γ grid, σ and η following Gibbs and Candès (2024), interval I = 72 h (§7.3).
- **Practical thresholds:** Δ ≥ 0.05; δ ≥ 0.05; D ≥ 0.03 (§10.2); P0 is informational and does not affect decisions (§10.4).

- **Number of storm windows** according to §6 (`scripts/count_storms.py`, from Dst only; `results/storm_counts.json`):

| Split | Moderate | Intense | Of which superintense |
|---|---|---|---|
| train | 203 | 80 | 8 |
| calib | 129 | 26 | 0 |
| test | 81 | 24 | 2 |

  The test period has 24 intense-storm windows, above the threshold of 10 in §6, so H1–H3 are decided formally. Calib contains no superintense storm, so the intervals were not calibrated on such a case.

**Steps planned after freezing, before the test evaluation** (results recorded in §13 as execution of the plan, not as deviations):

1. Compute-time estimate after a smoke test.
2. Tuning and R0 (§7.2, §9.1).
3. C-Mond class sizes on calib and class merging under the rule in §7.3.
4. P0 (§10.4).
5. Archiving the results of steps 1–4 on Zenodo, then E-main.

## References

- Echer, E., Gonzalez, W. D., Tsurutani, B. T. (2008). Interplanetary conditions leading to superintense geomagnetic storms (Dst ≤ −250 nT) during solar cycle 23. *Geophysical Research Letters*, 35. [doi.org/10.1029/2007GL031755](https://doi.org/10.1029/2007GL031755)
- Gibbs, I., Candès, E. (2021). Adaptive conformal inference under distribution shift. arXiv:2106.00170. [arxiv.org/abs/2106.00170](https://arxiv.org/abs/2106.00170)
- Gibbs, I., Candès, E. (2024). Conformal inference for online prediction with arbitrary distribution shifts. *Journal of Machine Learning Research*, 25(162), 1–36. [www.jmlr.org/papers/v25/22-1218.html](https://www.jmlr.org/papers/v25/22-1218.html)
- Gonzalez, W. D., Joselyn, J. A., Kamide, Y., Kroehl, H. W., Rostoker, G., Tsurutani, B. T., Vasyliunas, V. M. (1994). What is a geomagnetic storm? *Journal of Geophysical Research*, 99(A4), 5771–5792. [doi.org/10.1029/93JA02867](https://doi.org/10.1029/93JA02867)
- King, J. H., Papitashvili, N. E. (2005). Solar wind spatial scales in and comparisons of hourly Wind and ACE plasma and magnetic field data. *Journal of Geophysical Research*, 110, A02104. [doi.org/10.1029/2004JA010649](https://doi.org/10.1029/2004JA010649)
- Laperre, B., Amaya, J., Lapenta, G. (2020). Dynamic time warping as a new evaluation for Dst forecast with machine learning. arXiv:2006.04667. [arxiv.org/abs/2006.04667](https://arxiv.org/abs/2006.04667)
- Newell, P. T., Sotirelis, T., Liou, K., Meng, C.-I., Rich, F. J. (2007). A nearly universal solar wind–magnetosphere coupling function inferred from 10 magnetospheric state variables. *Journal of Geophysical Research*, 112, A01206. [doi.org/10.1029/2006JA012015](https://doi.org/10.1029/2006JA012015)
- NOAA (2020). Solar Cycle 25 Prediction Panel analysis: minimum between cycles 24 and 25 in December 2019; cycle 24 the weakest in 100 years. [weather.gov/news/201509-solar-cycle](https://weather.gov/news/201509-solar-cycle)
- O'Brien, T. P., McPherron, R. L. (2000). An empirical phase space analysis of ring current dynamics: Solar wind control of injection and decay. *Journal of Geophysical Research*, 105(A4), 7707–7719. [doi.org/10.1029/1998JA000437](https://doi.org/10.1029/1998JA000437)
- Sandu, I. C., Beșliu-Ionescu, D. (2026). Operational Dst forecasting across solar cycle using physics-motivated gradient boosting. *Frontiers in Astronomy and Space Sciences*. [www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2026.1835176/full](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2026.1835176/full)
- World Data Center for Geomagnetism, Kyoto, Nose, M., Iyemori, T., Sugiura, M., Kamei, T. et al. (2015). Geomagnetic Dst index. [doi.org/10.17593/14515-74000](https://doi.org/10.17593/14515-74000)
- Yu, Zou, Pulkkinen, Chen (2026). Stable multivariate functional time series prediction for major geomagnetic indices. arXiv:2606.14417. [arxiv.org/abs/2606.14417](https://arxiv.org/abs/2606.14417)
