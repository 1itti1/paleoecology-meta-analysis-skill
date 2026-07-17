# Paleoecology Meta-Analysis Skill

**Synthesize multi-site, multi-proxy paleoecological data with peer-review-grade statistical rigor.**

A hybrid meta-analysis toolkit for paleoecology and paleoclimate research: paleoecology-native synthesis (z-score, Bootstrap BCa, GAM, Monte Carlo ensembles) as the main route, with classical effect sizes (log response ratio, Hedges' d) as a conditional module triggered by data structure — not discipline preference.

**Compass points:** 🧭 dual-channel · 📊 synthesize · 📏 calibrate · 🛡️ validate · 📝 cite

## 🎯 What It Does

This skill is designed for the questions researchers face after collecting multi-site core data:

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

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/1itti1/paleoecology-meta-analysis-skill.git
cd paleoecology-meta-analysis-skill
pip install -r requirements.txt
```

Or use as a TRAE Skill — copy the skill folder into your TRAE skills directory:

```bash
# Linux / macOS
mkdir -p ~/.trae/skills
cp -R paleoecology-meta-analysis ~/.trae/skills/

# Windows PowerShell
New-Item -ItemType Directory -Force $HOME\.trae\skills | Out-Null
Copy-Item -Recurse -Force .\paleoecology-meta-analysis $HOME\.trae\skills\
```

### 2. Run A Smoke Test

Verify the modules import correctly and the core pipeline works:

```python
import sys; sys.path.insert(0, 'scripts')

from preprocessing import bam_age_ensemble, zscore_standardize
from continuous_proxy import standardize_continuous_proxy, composite_continuous_proxy
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

If you see the ensemble shape and z-scores without errors, the toolkit is ready.

### 3. Run A Full Analysis

**Scenario 1 — Validate a proxy against observations:**

```python
from effect_size import log_response_ratio, effect_size_bca, rmsep
from continuous_proxy import calibrate_continuous_proxy, cross_validate_calibration

# Calibrate proxy → climate variable
calib = calibrate_continuous_proxy(proxy_values, calib_x, calib_y)

# Quantify systematic bias
ratios = log_response_ratio(proxy_values, observed_values)
ci = effect_size_bca(proxy_values, observed_values, n_boot=10000)
precision = rmsep(proxy_values, observed_values)
```

**Scenario 2 — Synthesize multi-site time series:**

```python
from preprocessing import bam_age_ensemble
from continuous_proxy import composite_continuous_proxy

# Propagate age uncertainty through synthesis
result = composite_continuous_proxy(
    site_values, site_ages, time_grid,
    age_ensembles=age_ens, proxy_errors=errors,
    n_members=500
)
# result['uncertainty_band'] has 5%/50%/95% percentiles
```

**Scenario 3 — Attribute change to an event:**

```python
from scenarios import scenario3_human_attribution, build_indicators, multi_window_robustness

# Define your own indicator system (any region, any taxa)
indicators = build_indicators(data_df, {
    'crop': ['Oryza', 'Triticum'],
    'forest': ['Quercus', 'Pinus'],
    'disturbance': ['Artemisia', 'Chenopodiaceae'],
})

result = scenario3_human_attribution(before_data, after_data, event_year=-1)
robustness = multi_window_robustness(time_series, ages, event_year=-1, windows=[100, 50, 25])
```

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
      Three-layer uncertainty (age + calibration + sampling)
      Six validation strategies
      Peer-review defensibility checklist
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
  I -->|Fail| K[Check assumptions / try block bootstrap]
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

## 🧩 Scenario Selection Logic

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

## 🛡️ Statistical Rigor

### Assumption Checks (run before any synthesis)

- **Normality**: Shapiro-Wilk + Q-Q plot (Bootstrap is asymptotically robust for n>30)
- **Temporal independence**: AR1 coefficient + Durbin-Watson (use block bootstrap if violated)
- **Spatial independence**: Moran's I (cluster before synthesizing if violated)
- **Sample size**: n>20 (BCa minimum), n>30 (asymptotic normality)

### Three-Layer Uncertainty Propagation

- **Age layer**: Sample complete age-depth curves from BAM/Bacon posterior (preserve stratigraphic monotonicity)
- **Calibration layer**: Proxy-climate calibration residuals as normal noise, σ from RMSEP
- **Sampling layer**: Bootstrap resampling propagates naturally

### Six Validation Strategies

LOOCV / multi-method consistency (≥2 methods) / multi-window (100/50/25 yr) / dual-indicator system / external data comparison / sensitivity analysis

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

## 📂 Repository Layout

```
paleoecology-meta-analysis-skill/
├── SKILL.md                    # TRAE Skill entry (frontmatter + 8 sections)
├── README.md                   # This file
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
└── scripts/                    # 6 Python modules, 46 functions
    ├── preprocessing.py        # 7 functions: age ensembles, z-score, auto-clustering, bias presets
    ├── continuous_proxy.py     # 6 functions: standardize, calibrate, composite, uncertainty, CV
    ├── synthesis.py            # 10 functions: 5-method synthesis, Monte Carlo, LOESS
    ├── effect_size.py          # 7 functions: log RR, Hedges' d, BCa, RMSEP, LOOCV
    ├── scenarios.py            # 7 functions: 3-scenario orchestration, indicators, robustness
    └── validation.py           # 9 functions: assumption checks, block bootstrap, 3-layer
```

## 🛠️ Practical Notes

- **Python-only**: R is blocked by Smart App Control on the developer's machine. BAM (pure Python) replaces Bacon/Clam age models — its RMSE of 251 yr is comparable to Bacon's 198 yr (Kaufman 2020 validation).
- **REVEALS gap**: The REVEALS model (Sugita 2007) has no Python implementation. The toolkit uses z-score standardization as a fallback and documents this gap explicitly in `references/methodology_gaps.md`.
- **Optional dependencies**: `pygam` (GAM synthesis), `scikit-learn` (k-fold CV), `libpysal`+`esda` (Moran's I) are listed in requirements.txt but not strictly required. The toolkit gracefully degrades when they're absent.
- **Parameter defaults** match published literature: `n_members=500` (Kaufman 2020), `n_boot=10000` (Izdebski 2022), `n_splines=20` (GAM), `frac=0.2` (LOESS, Cleveland & Devlin 1988).
- **Start small**: Run the smoke test first. Verify the modules import. Then move to a single-core analysis before attempting multi-site synthesis.

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

Also cite the methodology literature your analysis relies on (see table above).

## 🚢 Release Copy

- **Repository description**: `A hybrid meta-analysis toolkit for paleoecology and paleoclimate: dual-channel (taxa + continuous proxy) synthesis with BAM age ensembles, Bootstrap BCa uncertainty, and peer-review-grade statistical rigor.`
- **Tagline**: `Synthesize multi-site, multi-proxy paleoecological data with peer-review-grade statistical rigor.`
- **Current release**: `v2.0.0 — Dual-channel architecture, multi-region generalization`
