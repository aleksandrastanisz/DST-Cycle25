# PROTOCOL: Dst forecasts and uncertainty-interval coverage during storms of solar cycle 25

**Version:** 0.4 (draft, not frozen), 2026-11-24

**Status:** draft. The protocol is frozen once the open items in §15 are closed, before the first training run. After freezing, any modification requires an entry in §13 (date, reason, impact).

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
2. **Model value versus input data quality.** OMNI data are processed after the fact and time-shifted to the bow-shock nose. Results are an upper bound for operational forecasting.
3. **Storm-time coverage versus conditioning on the outcome.** Storm windows are defined from observed Dst (§6), so storm-time coverage is conditional on the outcome. Windows include every storm phase, which weakens this effect but does not remove it.
4. **Shift between cycles versus the difficulty of storms themselves.** We therefore also report storm-time coverage of C-Mond and C-DtACI on cycle 24, out of the calibration sample (§7.3, M9). A drop relative to cycle 24 measures shift; coverage below nominal on its own does not.

## 4. Hypotheses

Hypotheses are decided only by the rules in §10. Every result, including null results, is reported. Each hypothesis is tested for two ML models (X, L) and two horizons (1 h, 6 h), giving 4 tests per hypothesis.

- **H1 (ML advantage shrinks with storm intensity).** The skill score of an ML model relative to persistence is lower in intense-storm windows than in quiet hours.
- **H2 (conditional coverage does not transfer to a new cycle).** C-Mond intervals calibrated on cycle 24 have coverage below the nominal 90% in intense-storm windows of cycle 25.
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

**Dst versions** according to the Kyoto page (as of 2026-11-24): final 1957–2020, provisional 2021/01–2025/06, real-time only afterwards. The download script reads these ranges from the page at download time and saves a copy of it.

### 5.2 Time convention

- `t_start` is the start of the hour whose mean the record describes.
- The value for the interval [t − 1 h, t) is known at time t. A forecast issued at time t uses only such values.
- The target of a forecast with horizon h is Dst for the interval [t + h − 1 h, t + h).
- Kyoto and OMNI may label hours differently. The offset is determined on 1995–2014 data by comparing Kyoto Dst with Dst in OMNI. We accept an offset with at least 99% agreement; if none reaches it, work stops until the cause is explained. At h = 1 h, a one-hour offset would leak the target into the input.

### 5.3 Splits (finite, frozen, on disk)

Data are downloaded once by `scripts/download_data.py`. The download date, Dst version ranges, hour offset and SHA-256 checksums of all files are stored in `data/MANIFEST.json`.

| Split | Period (UTC) | Cycle (approx.) | Dst version | File | Use |
|---|---|---|---|---|---|
| train | 1995-01-01 – 2009-12-31 | 23 | final | `data/processed/train.parquet` | training and tuning (§7.2) |
| calib | 2010-01-01 – 2019-12-31 | 24 | final | `data/processed/calib.parquet` | conformal quantiles, warm-up of adaptive methods, M9 reference, power analysis; no model selection |
| test | 2020-01-01 – 2025-06-30 | 25 | final (2020), provisional (2021 – Jun 2025) | `data/sealed/test.parquet` | single final evaluation |

Split boundaries are calendar dates, not exact cycle boundaries: train also contains the tail of cycle 22 (1995–1996), and cycle 24 began in late 2008. Each stage nevertheless covers mainly one cycle. Calibrating on all of cycle 24, rather than on 2015–2019, ensures storms from different phases of the cycle.

**The test split** is touched **only** in the final evaluation. It is not used for tuning, model selection, debugging or plots. Exception, computed from Dst alone before any model is run: the number of storm windows in the test period (§6), which the power analysis also needs (§10.4).

## 6. Storms

**Definition** (fixed in advance, from the Dst series alone):

1. Storm minimum: an hour t with `Dst(t) ≤ −50 nT` where `Dst(t)` is the lowest value in [t − 24 h, t + 24 h]. Ties go to the earliest hour.
2. Storm window: [t_min − 24 h, t_min + 48 h]. Overlapping windows are merged; the category of a merged window is set by its lowest minimum.
3. A window belongs to the split that contains its minimum and is clipped to that split's boundaries.
4. A forecast hour belongs to a window if its **target hour** lies in the window.
5. **Quiet blocks:** target hours outside all windows are divided into consecutive 27-day calendar blocks (one solar rotation), starting at the beginning of the split. A block with fewer than 72 quiet hours is merged into the previous one.
6. **Storm phases:** the main phase is the target hours from the start of the window up to and including the window's lowest minimum; the recovery phase is the target hours after that minimum. In windows merged from several storms, phases are defined relative to the lowest minimum.

**Categories:**

| Category | Minimum Dst [nT] | Role |
|---|---|---|
| quiet | blocks from item 5 | reference in H1 |
| moderate | −50 to −100 | always reported |
| intense | ≤ −100 | H1–H3 |
| extreme | ≤ −250 (subset of intense) | descriptive only |

**Minimum number of storms.** If the test period contains fewer than **10** intense-storm windows, H1–H3 are reported descriptively, without a decision. The count is determined before any model is run and recorded in §15.

## 7. Models and training

- **Inputs to every model:** only values known at time t (§5.2). Current Dst is an input because it is available operationally (real-time index). This makes persistence a demanding reference.
- **P:** `Dst(t+h) = Dst(t)`.
- **E (AK2):** `dDst*/dt = Q − Dst*/τ`, `Q = −4.4·(VBs − 0.5)` nT/h for VBs > 0.5 mV/m and 0 otherwise, `τ = 2.4·exp(9.74 / (4.69 + VBs))` h, `Dst* = Dst − 7.26·√Pdyn + 11` (Dst in nT, Pdyn in nPa). VBs in mV/m: `V [km/s] · Bs [nT] · 10⁻³`, where Bs = −Bz for Bz < 0 and 0 otherwise. Integration with a 1 h step from `Dst*(t)` over h steps. Solar wind is held at its value at time t, since future values are unknown. Constants are not refitted.
- **X:** XGBoost (regression, squared loss) on the feature set in §7.1, one model per horizon. Missing feature values are handled natively. Every feature must pass the causality test (§14). Fixed: `subsample = 0.8`, `colsample_bytree = 0.8`, so that the seed affects the model.
- **L:** 24 h window of V, Bz and By in GSM, \|B\|, density, dynamic pressure, VBs and Dst; standardized with train statistics; one network per horizon; a single output.
- **Every fitted transformation** (standardization, any imputers) is learned on train only.
- **Missing data:** the last known value fills a gap of at most 3 h. A longer gap means no forecast for that hour. The main evaluation uses only the common set of hours in which **all** four models produce a forecast and the target is known. Because solar-wind data gaps may cluster in large storms, the fraction of excluded hours is reported for each category (M10) and S5 is run (§9).
- **Seeds:** 5 seeds (42–46) for X and L. The seed controls initialization, data order and XGBoost subsampling. The point forecast is the mean of the 5 models; spread across seeds is reported separately.
- **Ensemble as the unit:** hypotheses concern the seed-averaged forecast, since that is the model one would deploy.

### 7.1 Features of model X (fixed in advance)

All features are computed only from values known at time t (§5.2); "lag k" means the value for the interval [t − k − 1 h, t − k).

| Group | Features | Rationale |
|---|---|---|
| Ring-current state | Dst at lags 0, 1, 2, 3 and 6 h; `Dst(t) − Dst(t − 1 h)`; `Dst(t) − Dst(t − 3 h)` | current state and rate of change (main phase or recovery) |
| Solar wind | V, Bz and By in GSM, \|B\|, density, Pdyn at lags 0, 1, 2 and 3 h | direct driving |
| Coupling | VBs [mV/m]; Newell coupling function `V^(4/3)·B_T^(2/3)·sin^(8/3)(θ/2)`, where `B_T = √(By² + Bz²)`, `θ = atan2(By, Bz)`; `√Pdyn` | energy injection (Burton; O'Brien and McPherron) and the pressure correction in Dst* |
| Driving history | 3 h and 6 h running means of VBs, Bz, Pdyn and the Newell function; 6 h and 12 h sums of VBs | energy loaded before time t |

44 features in total: 7 + 24 + 3 + 10. The list is frozen with the protocol; no features are added or removed on the basis of results, including validation results. Feature importance (SHAP) is reported descriptively only.

### 7.2 Tuning (equal budget)

Tuning only within train: fit on 1995–2005, validate on 2006–2009, MSE on validation as the criterion, separate seed 100. Each ML model gets 12 configurations:

| Model | Grid |
|---|---|
| X | max_depth ∈ {4, 6, 8} × learning_rate ∈ {0.03, 0.1} × min_child_weight ∈ {1, 10}; number of trees by early stopping (patience 50, max 5,000) |
| L | hidden size ∈ {32, 64, 128} × lr ∈ {10⁻³, 3·10⁻³} × layers ∈ {1, 2}; Adam, batch 256, early stopping (patience 5, max 100 epochs) |

The final model is trained on all of train (1995–2009) with the selected configuration and the number of trees or epochs from the best tuning run. If the optimum lies on the edge of the grid, the grid is not extended and this is reported with the results. Whether the configuration is shared across horizons or selected per horizon is decided in §15.

### 7.3 Intervals

Nonconformity score for all methods: `|Dst − forecast|`. Calibration residuals come from calib (models trained on train only).

- **C-split:** quantile of order ⌈(n + 1)·0.9⌉ / n of the calib residuals. The width is constant for a given model and horizon.
- **C-Mond:** as C-split, but with separate quantiles for classes of `Dst(t)` at issue time: > −30, −30 to −50, −50 to −100, ≤ −100 nT. The class depends on the issue time, not on the outcome. A class with fewer than 200 hours in calib is merged with its neighbour closer to zero; class sizes are reported.
- **C-DtACI:** Algorithm 1 of Gibbs and Candès (2024): k = 8 ACI experts with steps γ ∈ {0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128}, combined with exponential weights; η and σ set by the formulas in the paper for an interval I = 72 h (the storm-window length). Quantile of order 1 − α(t) of the calibration residuals.
- **C-ACI (S1):** a single expert, `α(t+1) = α(t) + γ·(0.1 − err(t))`, γ ∈ {0.005, 0.05}.
- **Rules shared by the adaptive methods:**
  - **Feedback delay:** a forecast issued at time t yields `err` only at t + h. The update at time t uses only forecasts whose target is already known.
  - Start: α = 0.1 (and equal expert weights) on 2010-01-01; runs through all of calib and on through test without reset.
  - When α ≤ 0 the interval is infinite and counted as covered; when α ≥ 1 it is empty and counted as not covered. The number of such hours is reported, and width is reported as a median.
- **Cycle-24 reference (M9):** C-Mond and C-DtACI computed on the two halves of calib by cross-fitting: quantiles from 2010–2014 applied to 2015–2019 and vice versa. This gives intense-storm coverage out of the calibration sample but within the calibration cycle.

## 8. Metrics

All computed on the common set of hours (§7), separately for each model, horizon and category from §6.

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
| M7 | hidden-persistence check: the shift k ∈ {0, …, h + 3} h that maximizes the correlation between forecast and observation within a storm window; median over windows and fraction of windows with k ≥ h (cf. Laperre et al. 2020) |
| M8 | per window: forecast minimum minus observed minimum [nT] and timing error of the minimum [h] |
| M9 | intense-storm coverage of C-Mond and C-DtACI on cycle 24, cross-fitted (§7.3); difference M2(test) − M9 |
| M10 | fraction of hours excluded from evaluation because of missing data, per category |
| M11 | dose–response curve: M1 and M2 computed for each window separately as a function of the window's minimum Dst; scatter plot and means in 50 nT bins of the minimum |
| M12 | point-forecast shift: M3(test) / M3(calib) per category; calib is out of the training sample for the models |
| M13 | coverage curve: M2 at nominal levels 50, 60, 70, 80, 90 and 95% (C-split, C-Mond, C-DtACI), separately for quiet hours and intense storms |

M1–M5 are also reported separately for storm phases (§6, item 6).

## 9. Experiments

| ID | Description | Role |
|---|---|---|
| R0 | pipeline correctness checks on 2006–2009 (tuning validation), §9.1 | prerequisite; no test data |
| P0 | power analysis on cycle 24 (§10.4) | prerequisite; no test data |
| E-main | P, E, X, L × h ∈ {1, 6} × {C-split, C-Mond, C-DtACI} on test | confirmatory (H1, H2, H3) |
| S1 | C-ACI with γ ∈ {0.005, 0.05} | sensitivity to the adaptation scheme |
| S2 | hour-weighted statistics instead of unit-weighted | sensitivity to aggregation |
| S3 | storm windows [t_min − 12 h, t_min + 24 h] instead of §6 | sensitivity to the window definition |
| S4 | M6 and an H1 analogue relative to model E | whether conclusions depend on the choice of reference |
| S5 | P and X on all hours for which they themselves produce a forecast (XGBoost handles missing values) | sensitivity to missing-data exclusions |
| S6 | leave-one-storm-out: every confirmatory statistic recomputed without each intense-storm window in turn | stability of decisions |
| X1 | storms of 10–11 May and 10–11 October 2024: M2, M4 and M8 | exploratory, descriptive only |
| X2 | models frozen after E-main, evaluated on real-time Dst, 2025-07-01 – 2025-12-31 | exploratory; closer to operational conditions |
| X3 | H1–H3 counterparts for the extreme category | descriptive only |
| X4 | one repeat of training L (seed 42, both horizons) with the same configuration | scale of run-to-run noise from GPU nondeterminism; descriptive only |
| A1 | X and L without any Dst inputs (X without the "ring-current state" group, L without the Dst channel) | ablation: whether the advantage over persistence comes from the solar wind or from extending current Dst |
| A2 | X without the "coupling" group | ablation: contribution of the coupling functions |
| A3 | X without the "driving history" group | ablation: contribution of energy loaded before time t |
| A4 | X on the "ring-current state" group only | ablation: purely autoregressive model, complement to A1 |

**Ablations A1–A4** use seeds 42–46, the configurations selected when tuning the full model (no re-tuning), the same data splits and all three interval methods. They therefore measure sensitivity at fixed hyperparameters. For each ablation we report M1, M2 and M7 per category and the difference from its E-main counterpart (same model, horizon and seeds), with a 95% bootstrap interval (§10.1).

### 9.1 Correctness checks (R0)

Run on the tuning validation data (2006–2009), before any contact with calib or test. Failure of any check means a bug to find, not a result.

1. **Positive control:** X and L must have lower MSE than P at both horizons. A model that does not beat persistence on data from the same cycle is most likely wired incorrectly.
2. **Target-leakage control:** X trained with the target added as a feature must reach MAE close to zero (< 1 nT). This confirms the pipeline can detect leakage when it exists.
3. **Negative control:** X trained on a target randomly shuffled within train must not beat a constant forecast (the train mean) by more than 1% in MSE.
4. **Model E:** correlation between forecast and observation > 0.8 at h = 1 h. This checks units (VBs in mV/m, Pdyn in nPa) and the integration.

**Analysis of S1–S5, A1–A4 and X1–X4** is descriptive, without hypothesis tests: point estimate and 95% bootstrap interval (§10).

**Run order** (most important for interpretation first): R0, P0, tuning, E-main, S6, A1, A4, S5, S2, S4, A2, A3, S1, S3, X4, X1, X3, X2.

## 10. Statistics and decision rules

### 10.1 Units and bootstrap

**The independent unit** is a storm window or a quiet block (§6), not an hour, because hours within one storm are strongly correlated.

**Bootstrap.** In each replicate, storm windows are resampled with replacement separately within each category, and quiet blocks are resampled likewise. One set of draws is shared by all models, horizons and interval methods, because comparisons are paired. 10,000 replicates, seed 2026. Percentile 95% intervals. The one-sided p-value is the fraction of replicates in which the statistic lies on or beyond the null boundary (0 for H1 and H3, 0.90 for H2).

**Correction.** Holm's method separately within each family: 4 tests per hypothesis (X, L × 1 h, 6 h), α = 0.05.

### 10.2 Rules

**H1.** For model v and horizon h: `Δ = M1(quiet) − M1(intense)`. H1 is supported for (v, h) when both:

- `Δ ≥ 0.05` (practical threshold),
- Holm-adjusted p < 0.05.

**H2.** For C-Mond on model v and horizon h: `cov = M2(intense)`. H2 is supported when both:

- `cov ≤ 0.85` (at least 5 percentage points below nominal),
- Holm-adjusted p < 0.05 for `cov < 0.90`.

M9 and the difference M2(test) − M9 are reported alongside; they do not enter the rule, but without them an H2 result is not interpreted as an effect of shift between cycles (§3, item 4).

**H3.** `D = M2(intense; C-DtACI) − M2(intense; C-Mond)`. H3 is supported when all of:

- `D ≥ 0.03`,
- Holm-adjusted p < 0.05 for `D > 0`,
- `M2(intense; C-DtACI) ≥ 0.85`.

The ratio of median widths of C-DtACI to C-Mond is also reported; it does not enter the rule.

### 10.3 Robustness

- **Few clusters.** A bootstrap over a dozen or so clusters can be anti-conservative. For H3, where the comparison is paired within each window, an exact sign-flip permutation test on the per-window differences `D_j` is also reported. The rule in §10.2 decides; any disagreement with the permutation test is flagged in the report.
- **Influence of a single storm (S6).** If removing one window changes the decision for any hypothesis, the decision remains formally valid but is reported as **fragile**, naming that storm.
- **Missing data.** If M10 in the intense category exceeds 10%, H1–H3 results are reported with a caveat and alongside S5.

### 10.4 Power analysis (P0, before freezing)

On cycle 24, with models trained on train: resample with replacement as many intense-storm windows as the test period contains (count from §6), and for effects equal to the thresholds in §10.2 estimate the probability that each hypothesis is supported (1,000 simulations). Report the minimum detectable effect at 80% power. If the thresholds are undetectable, they are changed **before** freezing and the reason is recorded in §15. Thresholds are not changed after freezing.

**Additionally (always):** M1–M13 for all models, horizons and categories, with 95% bootstrap intervals; spread of M1 and M3 across seeds; C-split as a reference.

**A null result is a result.** If a hypothesis is not supported, the report says so and shows the bootstrap distributions. The endpoint, thresholds, categories and storm definition must not be changed after seeing the results.

## 11. Threats to validity

| Threat | Mitigation |
|---|---|
| Different Dst versions in training and test | Stated in §3; X2 on real-time data as a check; re-evaluation once final Dst is published (out of scope) |
| OMNI data processed after the fact | Scope of conclusions (§12) |
| Leakage through an hour offset | Alignment check on 1995–2014 (§5.2) and feature causality test (§14) |
| Non-causal features or transformations fitted on all data | Causality test for every feature (§14); transformations fitted on train only; R0 checks (§9.1) |
| Features chosen to fit the result | Feature list frozen with the protocol (§7.1); no feature selection |
| Hypotheses formulated after reading papers with cycle-25 results (e.g. Yu et al. 2026: pooled coverage below nominal) | Disclosed in the paper; blinding covers evaluation of our models, not knowledge of the literature |
| The author knows how the 2024 storms unfolded | Blinding covers model evaluation, not knowledge of events; storm definitions and rules fixed before evaluation |
| Storm windows defined from observed Dst | §3; S3 tests sensitivity to the window definition |
| Few intense storms in the test period; bootstrap with few clusters | 10-storm threshold (§6); power analysis P0; permutation test for H3; extreme category descriptive only |
| A single storm (e.g. May 2024) dominates the statistics | Equal weighting of windows (§8); S6 and flagging of fragile decisions |
| Missing data clustered in storms | M10, 10% threshold and S5 |
| Unstable skill score in quiet hours | M1b as a complement; aggregating errors rather than ratios |
| Model E outside its fitted range in intense storms | Exploratory only; stated in §2 |
| Coverage below nominal may reflect storm difficulty rather than cycle shift | M9 as a cycle-24 reference (§3, item 4) |
| Hourly residuals are not exchangeable (autocorrelation) | The C-split and C-Mond guarantees are not exact even on calib; coverage on calib is reported as a check |
| Hyperparameters tuned on cycle 23 only | Limitation; equal tuning budget for X and L (§7.2) |
| Choice of step size in the adaptive method | C-DtACI tunes its own step; γ grid and interval I fixed in advance; C-ACI with fixed γ in S1 |
| Data revisions after download | Single download, SHA-256 checksums and date in `MANIFEST.json` |
| The ML advantage comes from extending current Dst rather than from the physics of driving | A1 and A4; M7 check |
| A bug in the statistics implementation | The §10 code is validated on synthetic data with a known effect before the final evaluation (§14) |
| Nondeterminism of GPU training | X4 as a noise measure; forecast as the mean of 5 seeds |

## 12. Scope of conclusions

Conclusions apply to: Dst forecasts at horizons of 1 h and 6 h, the OMNI and Kyoto data in §5, models P, E, X and L built in this project, trained on 1995–2009 and evaluated on 2020 – June 2025, the interval methods in §2, and the storm definitions in §6. They **do not apply** to operational forecasting from real-time L1 data, other indices (SYM-H, Kp), longer horizons, or models not listed in §2. X2 gives only an indication of operational conditions. Conclusions about extreme storms (≤ −250 nT) are descriptive.

**Comparison with the literature** (e.g. Sandu and Beșliu-Ionescu 2026, Yu et al. 2026) is descriptive only and takes place in the discussion, after the final evaluation. Differences in test periods, Dst versions and metrics mean the numbers are not directly comparable; it does not affect any decision.

## 13. Deviations log (changes after freezing)

| Date | What changed | Reason | Impact on analysis |
|---|---|---|---|
| | | | |

Changes made before freezing are visible in the Git history. Any use of test data before the final evaluation, including in code tests, is disclosed in this table, even if no model was evaluated on it.

## 14. Reproducibility

- Data: `scripts/download_data.py`, run once; `data/MANIFEST.json` with download date, URLs, Dst version ranges, hour offset, package versions and SHA-256 checksums.
- All model, interval and bootstrap parameters in `configs/*.yaml`. The report includes the config and commit hashes.
- Automated tests: OMNI parser and fill values, hour alignment, split disjointness, storm-window definition on synthetic data, causality of every feature (changing data after time t does not change the feature at time t), feedback delay in C-ACI and C-DtACI, agreement of the C-DtACI implementation with an example from Gibbs and Candès.
- The final evaluation script refuses to run if the SHA-256 of `data/sealed/test.parquet` does not match `MANIFEST.json` or the commit lacks the `protocol-v1.0` tag.
- The implementation of the bootstrap, Holm correction and rules in §10 is validated on synthetic data with a known effect (with and without an effect) before the final evaluation.
- Every run records the commit and an uncommitted-changes flag at start. E-main and ablation runs are launched only from a clean repository.
- XGBoost on CPU is reproducible with the same seed; LSTM training on GPU is not bitwise reproducible (X4 measures the scale of differences).
- `scripts/make_tables.py` regenerates every table and figure from saved forecasts, without manual steps.
- The frozen protocol, with the code and the R0 and P0 results, is archived on Zenodo before any evaluation on the test split.

## 15. Decisions prior to freezing

**Resolved:**

- **Horizons:** 1 h and 6 h with equal weight, with Holm correction over 4 tests per family.
- **Split:** mainly by cycles 23 / 24 / 25 (§5.3); the test period ends with the last month of provisional Dst (June 2025).
- **Nominal level:** 90%, because it yields more out-of-interval cases during storms than 95%, and therefore more power.
- **Unit of analysis:** storm window or quiet block; equal weighting of units.
- **Confirmatory methods:** C-Mond (H2) and C-DtACI (H3). C-split is not tested, because it does not promise conditional coverage (change from version 0.1).
- **Models:** all built from scratch in the project; model X features frozen in §7.1 (change from version 0.2, in which model X was based on the Sandu and Beșliu-Ionescu code).
- **Model E constants:** b = 7.26 nT·nPa^−1/2 and c = 11 nT, following O'Brien and McPherron (2000).

**Open (to close before freezing):**

- [ ] R0 result (correctness checks, §9.1)
- [ ] Kyoto-to-OMNI hour offset and agreement (from `MANIFEST.json`)
- [ ] OMNI column numbers and fill values checked against `omni2.text`
- [ ] η and σ formulas for C-DtACI and the γ grid copied from Gibbs and Candès (2024)
- [ ] C-Mond class sizes on calib and any class merging
- [ ] Tuning grids in §7.2 and whether the configuration is shared across horizons
- [ ] Number of intense-storm windows in the test period (§6)
- [ ] P0 result and final practical thresholds in §10.2 (currently Δ ≥ 0.05; cov ≤ 0.85; D ≥ 0.03)
- [ ] Compute-time estimate after a smoke test (tuning, E-main, ablations) on a GTX 1660
