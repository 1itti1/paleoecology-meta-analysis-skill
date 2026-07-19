# Paleoecology Meta-Analysis Skill

**[English](README.md)** | [简体中文](README_zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-2.1.0-green.svg)](https://github.com/1itti1/paleoecology-meta-analysis-skill)
[![DOI](https://img.shields.io/badge/DOI-cite-orange.svg)](#citation)

> Synthesize multi-site, multi-proxy paleoecological data with peer-review-grade statistical rigor.

A hybrid meta-analysis toolkit for paleoecology and paleoclimate research. It combines paleoecology-native synthesis (z-score, Bootstrap BCa, GAM, Monte Carlo ensembles) as the main route with classical effect sizes (log response ratio, Hedges' d) as a conditional module — triggered by data structure, not discipline preference.

**Compass points:** 🧭 dual-channel · 📊 synthesize · 📏 calibrate · 🛡️ validate · 📝 cite

---

## Table of Contents

- [What It Does](#-what-it-does)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Mental Model](#-mental-model)
- [Workflow Pipeline](#-workflow-pipeline)
- [Supported Data Types](#-supported-data-types)
- [Scenario Selection](#-scenario-selection)
- [Dual-Channel Architecture](#-dual-channel-architecture)
- [Hybrid R Bridge](#-hybrid-r-bridge)
- [Module Reference](#-module-reference)
- [Statistical Rigor](#-statistical-rigor)
- [Preservation Bias Presets](#-preservation-bias-presets)
- [Code Examples](#-code-examples)
- [Practical Notes](#-practical-notes)
- [Literature Foundation](#-literature-foundation)
- [Repository Layout](#-repository-layout)
- [FAQ](#-faq)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

---

## 🎯 What It Does

This toolkit is designed for the questions researchers face after collecting multi-site core data:

- How do I combine 3+ sediment cores with different time resolutions into one regional curve?
- Is my δDwax-reconstructed precipitation trustworthy? How large is the systematic bias?
- Did vegetation change after a policy/war/climate event, and can I attribute it to human activity?
- How do I propagate age uncertainty through every step of my synthesis?
- Will my statistical choices survive peer review?

It is especially useful when working with:

- Pollen, diatom, foraminifera, or other taxon-percentage data
- Continuous proxies like δDwax, brGDGTs, grain size, TOC, Mg/Ca
- Multi-site sediment cores with BAM/Bacon age models
- Before/after event comparison designs
- Karst, arid, tropical, lake, or marine environments

## ✨ Key Features

| Feature | Description |
|---|---|
| **Dual-channel architecture** | Separate pipelines for taxon-percentage data (pollen, diatoms) and continuous-value proxies (δDwax, brGDGTs), sharing a common statistical validation framework |
| **57 functions across 7 modules** | Preprocessing, continuous proxy, synthesis, effect size, scenarios, validation, R bridge — each function with literature-anchored docstrings |
| **Three scenario workflows** | Proxy validation, multi-site synthesis, event attribution — each with a complete method chain from raw data to peer-review-ready output |
| **Three-layer uncertainty propagation** | Age uncertainty (BAM/Bacon ensembles) + calibration uncertainty (RMSEP) + sampling uncertainty (Bootstrap), propagated jointly through 500-member Monte Carlo ensembles |
| **Five environment presets** | Preservation bias plugins for karst, arid, tropical, lake, and marine environments — auto-selected or manually overridden |
| **Literature-anchored parameters** | Every default value matches the published source: `n_members=500` (Kaufman 2020), `n_boot=10000` (Izdebski 2022), `n_splines=20` (GAM), `frac=0.2` (LOESS) |
| **Hybrid R bridge** | Optional R + metafor backend for classical meta-analysis (rma, Egger test, forest/funnel plots). Auto-detected; falls back to Python DerSimonian-Laird when R unavailable. No rpy2 dependency |
| **TRAE Skill compatible** | Drop into `.trae/skills/` for automatic loading in TRAE IDE sessions |

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/1itti1/paleoecology-meta-analysis-skill.git
cd paleoecology-meta-analysis-skill
pip install -r requirements.txt
```

Or use as a TRAE Skill:

```bash
# Linux / macOS
mkdir -p ~/.trae/skills
cp -R paleoecology-meta-analysis ~/.trae/skills/

# Windows PowerShell
New-Item -ItemType Directory -Force $HOME\.trae\skills | Out-Null
Copy-Item -Recurse -Force .\paleoecology-meta-analysis $HOME\.trae\skills\
```

### 2. Smoke Test

Verify the modules import and the core pipeline works:

```python
import sys; sys.path.insert(0, 'scripts')

from preprocessing import bam_age_ensemble, zscore_standardize
from continuous_proxy import standardize_continuous_proxy
import numpy as np

# Generate a small test age ensemble
depths = np.array([0, 10, 20, 30, 40])
ages = np.array([0, 500, 1000, 1500, 2000])
age_errors = np.array([20, 30, 40, 50, 60])
ens = bam_age_ensemble(depths, ages, age_errors, n_members=10)
print(f"Age ensemble shape: {ens['age_ensembles'].shape}")

# Standardize a continuous proxy
values = np.array([15.2, 14.8, 16.1, 13.5, 12.9, 14.0, 15.5])
std = standardize_continuous_proxy(values, method='zscore')
print(f"Z-scores: {std['standardized']}")
```

### 3. Run a Full Analysis

See [Code Examples](#-code-examples) for complete walkthroughs of all three scenarios.

## 🧠 Mental Model

```
mindmap
  root((Paleoecology Meta-Analysis))
    🧭 Dual-Channel
      Taxa channel
        Pollen, diatom, foraminifera
        z-score → SCC/DCC/CPS/PAI/GAM
      Continuous channel
        δDwax, brGDGTs, grain size
        Standardize → Calibrate → Composite
    📊 Synthesize
      BAM age ensembles (500 members)
      Monte Carlo uncertainty propagation
      LOESS trend visualization
      Multi-method cross-validation
    📏 Calibrate
      OLS / SMA regression
      LOOCV / k-fold validation
      RMSEP prediction error
      Dual-proxy comparison
    🛡️ Validate
      Normality / independence checks
      Three-layer uncertainty
      Six validation strategies
      Peer-review defensibility
    📝 Cite
      Literature-anchored parameters
      Docstring source attribution
      BibTeX / CITATION.cff
```

## 🧭 Workflow Pipeline

```
flowchart LR
  A[Raw core data] --> B{Proxy type?}
  B -->|Taxa %| C[z-score standardize]
  B -->|Continuous| D[Standardize + calibrate]
  C --> E[Spatial align / cluster]
  D --> E
  E --> F[Synthesize: SCC/GAM/CPS or weighted mean]
  F --> G[Monte Carlo ensemble 500 members]
  G --> H[Uncertainty band 5/50/95%]
  H --> I{Validation}
  I -->|Pass| J[Peer-review-ready output]
  I -->|Fail| K[Check assumptions / block bootstrap]
  K --> F
```

## 📥 Supported Data Types

| Input | Channel | Behavior |
|---|---|---|
| Pollen / diatom / foraminifera percentages | Taxa | z-score → harmonize names → check preservation bias → synthesize |
| δDwax / brGDGTs / Mg/Ca / Uk37 | Continuous | Standardize → calibrate to climate variable → composite |
| Grain size / TOC | Continuous | Standardize (z-score or robust) → composite without calibration |
| BAM age ensembles | Shared | 500-member Monte Carlo propagation through all downstream steps |
| Bacon/Clam age output | Shared | `consume_bacon_ages()` imports existing age models |
| Multi-site coordinates | Shared | `spatial_clustering()` auto-selects grid or distance clustering |

## 🧩 Scenario Selection

The toolkit selects methods based on **data structure**, not discipline preference:

| Data structure | Scenario | Effect size? | Recommended method chain |
|---|---|---|---|
| Paired comparison (proxy vs truth) | Scenario 1: Proxy validation | ✅ Yes | z-score → LOOCV → log response ratio → BCa → RMSEP |
| Multi-site time series stacking | Scenario 2: Regional synthesis | ❌ No | BAM → z-score → spatial align → SCC/GAM/CPS → 500-member ensemble → LOESS |
| Before/after event comparison | Scenario 3: Event attribution | Optional | z-score + indicators → BCa difference test → multi-window robustness |

**Why not always use effect sizes?** Classical meta-analysis (Hedges 1999) requires paired treatment/control structure. Time-series stacking violates independence assumptions — using effect sizes there would be statistically indefensible. The toolkit enforces this boundary automatically.

## 📊 Dual-Channel Architecture

| Channel | Proxies | Core module | Standardization | Synthesis |
|---|---|---|---|---|
| Taxa | Pollen, diatom, foraminifera, spores, macrofossils | `preprocessing.py` + `synthesis.py` | z-score | SCC / DCC / CPS / PAI / GAM |
| Continuous | δDwax, brGDGTs, grain size, TOC, Mg/Ca, Uk37 | `continuous_proxy.py` | z-score / minmax / robust | Weighted mean + Monte Carlo |

Both channels share `effect_size.py`, `validation.py`, and `scenarios.py` — statistical rigor is consistent regardless of proxy type.

## 🔗 Hybrid R Bridge

Classical meta-analysis functions (random effects pooling, heterogeneity statistics, publication bias tests, forest/funnel plots) are implemented through a **hybrid backend** in `r_bridge.py`:

| Backend | Trigger | Functions | Estimators |
|---|---|---|---|
| **metafor** (R) | R 4.0+ + metafor installed | `rma_random_effects`, `meta_regression`, `egger_test`, `forest_plot`, `funnel_plot`, `subgroup_analysis` | REML, ML, DL, EB, HS |
| **Python** (fallback) | R unavailable or metafor not installed | `rma_random_effects` (DerSimonian-Laird only) | DL |

**How it works:**

```
Python (NumPy arrays) → JSON → Rscript subprocess → metafor::rma() → JSON → Python (Dict)
```

No rpy2 dependency. The bridge auto-detects R at first call via `check_r_environment()` and caches the result. Use `get_backend()` to check which backend is active.

**Install R + metafor (optional, recommended):**

```r
# In R console:
install.packages("metafor")
```

When R is available, you get the full metafor feature set (REML estimation, Hartung-Knapp correction, I² heterogeneity, Egger's test, publication-quality forest/funnel plots). When R is absent, the toolkit gracefully falls back to a Python DerSimonian-Laird implementation — your analysis still runs, just with fewer estimator options.

**Example:**

```python
import sys; sys.path.insert(0, 'scripts')
import numpy as np
from r_bridge import rma_random_effects, forest_plot, check_r_environment

# Check backend
env = check_r_environment()
print(f"Backend: {env['metafor_ready'] and 'metafor' or 'python'}")

# Random effects meta-analysis
effect_sizes = np.array([0.35, 0.42, 0.28, 0.51, 0.39])
variances = np.array([0.08, 0.12, 0.06, 0.10, 0.09])
result = rma_random_effects(effect_sizes, variances, method='REML', knha=True)
print(f"Pooled effect: {result['pooled_effect']:.4f}")
print(f"I²: {result['I2']:.1f}%  τ²: {result['tau2']:.4f}")
print(f"Backend used: {result['backend']}")

# Forest plot (requires metafor)
forest_plot(effect_sizes, variances,
            study_labels=['Site A', 'Site B', 'Site C', 'Site D', 'Site E'],
            output_path='forest.png')
```

## 📦 Module Reference

| Module | Functions | Key functions | Literature basis |
|---|---|---|---|
| `preprocessing.py` | 7 | `bam_age_ensemble`, `zscore_standardize`, `resample_to_grid`, `spatial_clustering` (auto), `harmonize_names`, `record_preservation_bias` (presets) | Comboul 2014, Kaufman 2020, Izdebski 2022 |
| `continuous_proxy.py` | 6 | `standardize_continuous_proxy`, `calibrate_continuous_proxy`, `composite_continuous_proxy`, `propagate_continuous_uncertainty`, `cross_validate_calibration`, `proxy_comparison` | Kaufman 2020, Roberts 2018 |
| `synthesis.py` | 10 | `scc_composite`, `dcc_composite`, `cps_composite`, `pai_composite`, `gam_composite`, `monte_carlo_ensemble`, `loess_trend` | Kaufman 2020, Marlon 2008 |
| `effect_size.py` | 7 | `log_response_ratio`, `hedges_d`, `effect_size_bca`, `rmsep`, `loocv` | Hedges 1999, Izdebski 2022 |
| `scenarios.py` | 7 | `scenario1_proxy_validation`, `scenario2_multi_site_synthesis`, `scenario3_human_attribution`, `build_indicators` | All 7 sources |
| `validation.py` | 9 | `check_normality_bootstrap`, `check_temporal_independence`, `check_spatial_independence`, `propagate_three_layer_uncertainty` | Izdebski 2022, Kaufman 2020 |
| `r_bridge.py` | 11 | `rma_random_effects`, `meta_regression`, `egger_test`, `forest_plot`, `funnel_plot`, `subgroup_analysis` | Viechtbauer 2010, Hartung & Knapp 2001, Egger 1997 |

**Module dependency chain:**

```
preprocessing ─┬─→ scenarios
continuous_proxy┘
synthesis ─────┘
effect_size ───┘
r_bridge ───────┘  (optional: enhances effect_size when R+metafor available)
validation ────┘  (called by all scenarios)
```

## 🛡️ Statistical Rigor

### Assumption Checks (run before any synthesis)

| Assumption | Test | Action if violated |
|---|---|---|
| Normality | Shapiro-Wilk + Q-Q plot | Use Bootstrap (asymptotically robust for n>30) |
| Temporal independence | AR1 coefficient + Durbin-Watson | Switch to block bootstrap |
| Spatial independence | Moran's I | Cluster sites before synthesizing |
| Sample size | n>20 (BCa), n>30 (asymptotic) | Report as preliminary / collect more data |

### Three-Layer Uncertainty Propagation

```
Age layer          Calibration layer      Sampling layer
    │                    │                      │
    ▼                    ▼                      ▼
BAM/Bacon          RMSEP as σ            Bootstrap
posterior          for normal noise      resampling
    │                    │                      │
    └────────┬───────────┘                      │
             ▼                                   │
      500-member Monte Carlo ensemble ◄─────────┘
             │
             ▼
      5%/50%/95% uncertainty band
```

### Six Validation Strategies

1. **LOOCV** — Leave-one-out cross-validation of calibration models
2. **Multi-method consistency** — Run ≥2 synthesis methods, compare results
3. **Multi-window robustness** — Test with 100/50/25-year windows
4. **Dual-indicator system** — Cross-validate with independent proxies
5. **External data comparison** — Compare with instrumental/independent records
6. **Sensitivity analysis** — Vary key parameters, assess result stability

## 🔌 Preservation Bias Presets

`record_preservation_bias()` ships with 5 environment presets as plugins:

| Preset key | Environment | Sensitive taxa | Tolerant taxa |
|---|---|---|---|
| `pollen-karst` | Karst alkaline soil | Ericaceae | Poaceae, Cyperaceae |
| `pollen-arid` | Arid zone | Pteridophyta | Chenopodiaceae, Artemisia |
| `pollen-tropical` | Tropical oxidation | Moraceae, Melastomataceae | Poaceae |
| `diatom-lake` | Lake diatom | Fragilaria, Eunotia | Aulacoseira, Stephanodiscus |
| `foraminifera-marine` | Marine foraminifera | Globigerinoides | Globorotalia |

Override with `sensitive_taxa` and `tolerant_taxa` parameters for fully custom environments.

## 💻 Code Examples

### Scenario 1: Proxy Validation

Validate a δDwax-reconstructed precipitation against instrumental observations:

```python
import sys; sys.path.insert(0, 'scripts')
import numpy as np

from effect_size import log_response_ratio, effect_size_bca, rmsep
from continuous_proxy import calibrate_continuous_proxy, cross_validate_calibration

# --- Input data ---
proxy_values = np.array([...])      # δDwax-reconstructed δDp
observed_values = np.array([...])   # Instrumental δDp
calib_x = np.array([...])           # Calibration set: δDwax
calib_y = np.array([...])           # Calibration set: δDp

# --- Step 1: Calibrate proxy → climate variable ---
calib = calibrate_continuous_proxy(proxy_values, calib_x, calib_y, regression='ols')
print(f"Calibration R²={calib['r2']:.3f}, RMSEP={calib['rmsep']:.2f}")

# --- Step 2: Cross-validate the calibration model ---
cv = cross_validate_calibration(calib_x, calib_y, method='loocv')
print(f"LOOCV RMSEP={cv['rmsep']:.2f}, R²={cv['r2']:.3f}")

# --- Step 3: Quantify systematic bias ---
ratios = log_response_ratio(proxy_values, observed_values)
ci = effect_size_bca(proxy_values, observed_values, n_boot=10000)
precision = rmsep(proxy_values, observed_values)
print(f"Mean log response ratio: {np.mean(ratios):.4f}")
print(f"95% BCa CI: ({ci[0]:.4f}, {ci[1]:.4f})")
print(f"RMSEP: {precision:.2f}")
```

### Scenario 2: Multi-Site Synthesis

Combine 3 lake cores into one regional vegetation trend:

```python
from preprocessing import bam_age_ensemble, zscore_standardize
from continuous_proxy import standardize_continuous_proxy, composite_continuous_proxy
import numpy as np

# --- Input: 3 sites with different resolutions ---
time_grid = np.arange(-2000, 0, 20)  # 2000 BP to present, 20-yr bins

# Site A: high-resolution, has age ensembles
site_a_ages = np.array([...])
site_a_values = np.array([...])
site_a_ens = bam_age_ensemble(site_a_depths, site_a_ages, site_a_errors, n_members=500)

# Site B & C: similar structure...
# (stack into arrays: site_values (n_sites, n_depths), site_ages (n_sites, n_depths))

# --- Step 1: Standardize each site ---
std_a = standardize_continuous_proxy(site_a_values, method='zscore')

# --- Step 2: Composite with uncertainty propagation ---
result = composite_continuous_proxy(
    site_values,     # (3, n_depths)
    site_ages,       # (3, n_depths)
    time_grid,       # (n_bins,)
    age_ensembles=site_a_ens['age_ensembles'],  # shared age model
    proxy_errors=np.array([0.5, 0.7, 0.6]),     # per-site RMSEP
    n_members=500,
)

# --- Step 3: Extract uncertainty band ---
band = result['uncertainty_band']
print(f"Median curve shape: {band['median'].shape}")
print(f"90% CI at 1000 BP: [{band['lower'][50]:.2f}, {band['upper'][50]:.2f}]")
```

### Scenario 3: Event Attribution

Test whether vegetation changed after a historical event (e.g., policy reform in 1726 AD):

```python
from scenarios import build_indicators, scenario3_human_attribution, multi_window_robustness

# --- Step 1: Define your own indicator system ---
# (fully user-customized — any region, any taxa)
indicators = build_indicators(pollen_df, {
    'crop': ['Oryza', 'Triticum', 'Hordeum'],
    'pasture': ['Poaceae', 'Cyperaceae'],
    'forest': ['Quercus', 'Pinus', 'Castanopsis'],
    'disturbance': ['Artemisia', 'Chenopodiaceae'],
}, agg_func='sum')

# --- Step 2: Split before/after event ---
event_year = 1726
before_data = indicators.loc[indicators.index < event_year].values
after_data = indicators.loc[indicators.index >= event_year].values

# --- Step 3: Test difference ---
result = scenario3_human_attribution(
    before_data, after_data, event_year=event_year, n_boot=10000
)
for r in result['results']:
    print(f"{r['indicator']}: diff={r['difference']:.3f}, "
          f"CI=({r['ci'][0]:.3f}, {r['ci'][1]:.3f}), p={r['p_value']:.4f}")

# --- Step 4: Multi-window robustness ---
robustness = multi_window_robustness(
    time_series, ages, event_year=event_year, windows=[100, 50, 25]
)
print(f"Robust across windows: {robustness['robust']}")
```

## 🛠️ Practical Notes

- **Hybrid R bridge**: R + metafor are auto-detected at runtime. When available, classical meta-analysis uses metafor (REML, Egger test, forest plots). When absent, a Python DerSimonian-Laird fallback is used. BAM (pure Python) always replaces Bacon/Clam age models — its RMSE of 251 yr is comparable to Bacon's 198 yr (Kaufman 2020 validation).
- **REVEALS gap**: The REVEALS model (Sugita 2007) has no Python implementation. The toolkit uses z-score standardization as a fallback and documents this gap explicitly in `references/methodology_gaps.md`.
- **Optional dependencies**: `pygam` (GAM synthesis), `scikit-learn` (k-fold CV), `libpysal`+`esda` (Moran's I), R + `metafor` (classical meta-analysis) are listed in requirements.txt but not strictly required. The toolkit gracefully degrades when they're absent.
- **Parameter defaults** match published literature: `n_members=500` (Kaufman 2020), `n_boot=10000` (Izdebski 2022), `n_splines=20` (GAM), `frac=0.2` (LOESS, Cleveland & Devlin 1988).
- **Start small**: Run the smoke test first. Verify the modules import. Then move to a single-core analysis before attempting multi-site synthesis.
- **Cross-module imports**: `scenarios.py` uses `sys.path.insert` to import other modules. If using modules independently, add the `scripts/` directory to your Python path.

## 📚 Literature Foundation

Every function's docstring cites its literature source. Parameters are named to match the original publications.

| Paper | Core method | Implementation |
|---|---|---|
| Izdebski 2022 (Nat Ecol Evol) | z-score → Bootstrap 10000× BCa → Bayesian GAM | `zscore_standardize()`, `effect_size_bca()` |
| Kaufman 2020 (Sci Data) | 5 methods (SCC/DCC/GAM/CPS/PAI) + 500-member ensemble | `scc_composite()`, `gam_composite()`, `monte_carlo_ensemble()` |
| Marlon 2008 (Nat Geosci) | Charcoal flux → LOESS smoothing | `loess_trend()` |
| Power 2008 (Clim Dyn) | z-score + regional synthesis | `zscore_standardize()`, `spatial_clustering()` |
| Roberts 2018 (Sci Rep) | REVEALS pollen→land cover | Gap documented |
| Hedges 1999 (Ecology) | log response ratio ln(X_T/X_C) | `log_response_ratio()`, `hedges_d()` |
| Lajeunesse 2009 (Am Nat) | Phylogenetic meta-analysis | Reference only |
| Comboul 2014 | BAM age model | `bam_age_ensemble()` |
| Cleveland & Devlin 1988 | LOESS locally weighted regression | `loess_trend()` |
| Sugita 2007 | REVEALS model | Gap documented |
| Viechtbauer 2010 (J Stat Soft) | metafor: meta-analysis framework | `rma_random_effects()`, `forest_plot()` |
| Hartung & Knapp 2001 (Biometrics) | Hartung-Knapp adjustment | `rma_random_effects(knha=True)` |
| Higgins & Thompson 2002 (JRSS-A) | I² heterogeneity statistic | `rma_random_effects()` |
| Egger et al. 1997 (BMJ) | Publication bias test | `egger_test()` |

## 📂 Repository Layout

```
paleoecology-meta-analysis-skill/
├── SKILL.md                    # TRAE Skill entry (frontmatter + 8 sections)
├── README.md                   # English documentation (this file)
├── README_zh-CN.md             # Chinese documentation
├── LICENSE                     # MIT
├── CITATION.cff                # Academic citation metadata
├── requirements.txt            # Python dependencies
├── .gitignore
├── .gitattributes
├── references/                 # 7 methodology reference docs
│   ├── preprocessing.md        # Ch.3: BAM, z-score, spatial align, bias presets
│   ├── synthesis_methods.md    # Ch.4: SCC/DCC/CPS/PAI/GAM, BCa, LOESS, ensembles
│   ├── effect_size.md          # Ch.5: log response ratio, Hedges' d, boundaries
│   ├── scenarios.md            # Ch.6: 3 scenarios, decision flowchart, dual-channel
│   ├── validation.md           # Ch.7: assumption tests, 3-layer uncertainty, 6 strategies
│   ├── python_toolchain.md     # Ch.8: 12 verified tools, version compatibility
│   └── methodology_gaps.md     # Ch.9-10: 4 gaps, peer-review checklist
└── scripts/                    # 7 Python modules, 57 functions
    ├── preprocessing.py        # 7 functions: age ensembles, z-score, auto-clustering
    ├── continuous_proxy.py     # 6 functions: standardize, calibrate, composite, CV
    ├── synthesis.py            # 10 functions: 5-method synthesis, Monte Carlo, LOESS
    ├── effect_size.py          # 7 functions: log RR, Hedges' d, BCa, RMSEP, LOOCV
    ├── scenarios.py            # 7 functions: 3-scenario orchestration, indicators
    ├── validation.py           # 9 functions: assumption checks, block bootstrap, 3-layer
    └── r_bridge.py             # 11 functions: R+metafor bridge (rma, Egger, forest, funnel)
```

## ❓ FAQ

<details>
<summary><b>Can I use this toolkit with marine sediment cores?</b></summary>

Yes. The toolkit is region-agnostic. For marine cores, use `proxy_type='continuous'` for proxies like Mg/Ca or Uk37, and select the `foraminifera-marine` preservation bias preset if working with foraminifera assemblages. The BAM age model works with any sediment type.
</details>

<details>
<summary><b>Does this toolkit support R meta-analysis packages?</b></summary>

Yes, since v2.1. The `r_bridge.py` module provides a hybrid backend: when R + metafor are detected, classical meta-analysis functions (rma, Egger test, forest/funnel plots, meta-regression) use metafor via subprocess + JSON. When R is unavailable, the toolkit falls back to a Python DerSimonian-Laird implementation. No rpy2 dependency required. Run `check_r_environment()` to see which backend is active.
</details>

<details>
<summary><b>How do I import Bacon age model output?</b></summary>

Use `consume_bacon_ages(bacon_output_path)` in `preprocessing.py`. It loads Bacon's age ensemble output (`.txt` or `.csv`, each column is an ensemble member) and returns a standardized dict with `age_ensembles`, `depths`, and `n_members`.
</details>

<details>
<summary><b>What if my data violates independence assumptions?</b></summary>

The `validation.py` module provides `check_temporal_independence()` (AR1 + Durbin-Watson) and `check_spatial_independence()` (Moran's I). If violated, use `block_bootstrap()` instead of standard Bootstrap, and `spatial_clustering()` to group nearby sites before synthesis.
</details>

<details>
<summary><b>Can I add my own preservation bias preset?</b></summary>

Yes. Pass `sensitive_taxa` and `tolerant_taxa` as lists to `record_preservation_bias()`. You can also extend the `PRESERVATION_BIAS_PRESETS` dictionary in `preprocessing.py` to register a new preset permanently.
</details>

<details>
<summary><b>What's the difference between the two channels?</b></summary>

The **taxa channel** handles percentage data (pollen, diatoms, foraminifera) — it uses z-score standardization and five synthesis methods (SCC/DCC/CPS/PAI/GAM). The **continuous channel** handles single-value proxies (δDwax, brGDGTs) — it adds calibration regression, cross-validation, and uses weighted-mean compositing with Monte Carlo ensembles. Both share the same effect-size, validation, and scenario modules.
</details>

## 🗺️ Roadmap

- [x] v1.0 — Initial release: 5 modules, 40 functions, karst pollen focus
- [x] v2.0 — Generalization: dual-channel architecture, 6 modules, 46 functions, multi-proxy multi-region
- [x] v2.1 — Hybrid R bridge: metafor integration (rma, Egger, forest/funnel), 7 modules, 57 functions
- [ ] v2.2 — Add example datasets and Jupyter notebook tutorials
- [ ] v2.3 — PyMC Bayesian GAM implementation (replacing PyGAM for uncertainty quantification)
- [ ] v3.0 — REVEALS model Python port (collaboration welcome)
- [ ] v3.1 — Interactive web dashboard for visualization

## 🤝 Contributing

Contributions are welcome! Areas where help is especially needed:

- **REVEALS model Python implementation** — The biggest methodology gap (Sugita 2007)
- **Bayesian GAM** — PyMC-based replacement for PyGAM, with full posterior uncertainty
- **Test datasets** — Example sediment core data (anonymized) for tutorials
- **Additional language translations** — README in Japanese, German, French, Spanish
- **Bug reports and feature requests** — Open an issue on GitHub

To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-function`)
3. Ensure all functions have literature-cited docstrings
4. Verify with `python -m py_compile scripts/*.py`
5. Submit a pull request

## 📄 License

Released under the [MIT License](LICENSE).

## 📝 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{paleoecology_meta_analysis_skill_2026,
  title = {Paleoecology Meta-Analysis Skill},
  author = {paleoecology-research},
  version = {2.0.0},
  date = {2026-07-17},
  license = {MIT},
  url = {https://github.com/1itti1/paleoecology-meta-analysis-skill}
}
```

Also cite the methodology literature your analysis relies on (see [Literature Foundation](#-literature-foundation)).

## 🚢 Release Copy

- **Repository description**: `A hybrid meta-analysis toolkit for paleoecology and paleoclimate: dual-channel (taxa + continuous proxy) synthesis with BAM age ensembles, Bootstrap BCa uncertainty, and peer-review-grade statistical rigor.`
- **Tagline**: `Synthesize multi-site, multi-proxy paleoecological data with peer-review-grade statistical rigor.`
- **Current release**: `v2.1.0 — Hybrid R bridge, metafor integration`
