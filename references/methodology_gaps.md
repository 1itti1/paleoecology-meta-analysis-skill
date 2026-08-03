# Methodological gaps and reporting boundaries

## Chronology

The bundled perturbation helper is not a chronology model. Radiocarbon
calibration, reservoir correction, contamination, sedimentation priors, and
age–depth covariance must come from an archive-specific model or external
backend.

## Compositional proxies

Percentages are constrained by a denominator and may be affected by closure,
zero inflation, unequal counts, taxonomic resolution, pollen production,
transport, aquatic/local signals, and preservation. z-score standardization does
not correct these mechanisms. Retain counts and use an appropriate count or
compositional model when they affect the estimand.

## REVEALS and source-area models

REVEALS requires taxon-specific relative pollen productivity, dispersal, and
source-area assumptions. This skill does not implement REVEALS. A z-score of
percentages can be a descriptive sensitivity index only and must not be called
a land-cover reconstruction.

## Causal attribution

Historical population, land use, village, phytolith, climate, and pollen data
can support temporal consistency and association analyses. A before/after
contrast alone does not identify a counterfactual. Consider controls, climate
covariates, lags, spatial effects, serial correlation, and multiple-testing
correction before using causal language.

## Spatial synthesis

Clustering and weighted means are not substitutes for a spatial hierarchical
model. Coordinate distance, pollen source area, site selection, and unequal
record density can all affect regional estimates. Report site count per bin and
perform site-weight and spatial-neighborhood sensitivity analyses.

## Reporting checklist

- state the estimand and unit of replication;
- preserve raw data and preprocessing metadata;
- identify the chronology source and uncertainty shape;
- distinguish measurement error, chronology uncertainty, bootstrap intervals,
  and Bayesian credible intervals;
- report missingness, extrapolation policy, effective site count, random seed,
  and backend;
- separate descriptive association from causal claims;
- include archive-specific limitations and alternative explanations.
