# Benchmark Effective Dimensionality Under Algorithm-Portfolio Variation

Reproducibility materials for the manuscript **“Benchmark Effective Dimensionality Under Algorithm-Portfolio Variation”** by Huynh Anh Khiem (ORCID: 0009-0007-7210-174X).

This repository contains public, analysis-ready materials used to verify the paper’s main effective-dimensionality and algorithm-portfolio results. The original UCI and OpenML source datasets are public and are not redistributed here.

## Contents

- `analysis/reproduce_results.py` — decodes the public analysis-ready matrices and checks the main reported diagnostics.
- `analysis/requirements.txt` — minimal Python dependencies.
- `data/encoded/` — compact analysis-ready primary score matrices.
- `data/openml_cc18_portfolio_k84.csv` — frozen K=84 configuration portfolio.
- `data/openml_cc18_family_map.csv` — mapping from configurations to algorithm families.
- `data/openml_cc18_portfolio_stability.csv` — saved finite-portfolio stability results.
- `data/openml_cc18_split_half_reliability.csv` — saved split-half reliability results.
- `data/uci_classification_linked_pairs.csv` and `data/uci_classification_provenance_clusters.csv` — provenance-audit inputs.
- `protocol/FROZEN_PROTOCOL_CC18_K84.json` — frozen CC18 K=84 protocol.
- `verified_reported_results.json` — exact machine-readable results from the full saved-output rerun used to verify the manuscript.

## Reproduce the public checks

```bash
python -m pip install -r analysis/requirements.txt
python analysis/reproduce_results.py
```

The script writes `public_reproduction_check.json`. It independently recomputes the primary participation-ratio estimates and the main CC18 portfolio diagnostics from the public analysis-ready matrices, then compares them with the exact values retained in `verified_reported_results.json`.

The CC18 balanced-accuracy matrix is quantized to 1e-6 for compact public storage. This changes the participation-ratio estimate only below the reporting precision used in the manuscript. Exact manuscript values from the full saved-output rerun are preserved in `verified_reported_results.json`.

## Scope of the public package

The public package is intended to make the analysis layer inspectable without redistributing the raw UCI/OpenML datasets or the much larger intermediate fold-level cache. It includes enough material to verify the paper’s main effective-dimensionality result, the K=36 versus K=84 change, the portfolio-sufficiency criterion (K*=48 at tau=0.90 and q=0.10), family-centroid robustness, leave-one-family-out robustness, and split-half reliability.

The exact saved-output summaries for permutation references, metric sensitivity, held-out subspace reconstruction, cap sensitivity, and the provenance-response audit are also retained in `verified_reported_results.json`.
