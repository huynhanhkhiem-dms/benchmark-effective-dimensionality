# Benchmark Effective Dimensionality and Disjoint Algorithm Portfolios

Public research data and reproducibility materials for Huynh Anh Khiem (ORCID: 0009-0007-7210-174X), Faculty of Information Technology, Ton Duc Thang University.

The revised manuscript is titled **Disjoint Algorithm Portfolios and the Reliability of Benchmark Geometry: Evidence from Representative-Task Selection**. Its primary new results concern disjoint algorithm portfolios, task-neighbor retention, and representative-task subset transfer. The original participation-ratio estimator is **not** claimed as a novel invention.

## Run the verification

Prerequisites: Python 3.11, numpy 2.3.5, pandas 2.2.3. From the repository root:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/reproduce_results.py
python analysis/reproduce_submission.py --repo . --reference results/reference_20261008.json --output recomputed_submission_results.json
```

The latter script:
- decodes the **actual CC18 analysis-ready balanced-accuracy matrix (72 tasks x 84 configurations)**, stored in `data/encoded/` as compressed base64 with 1e-6 score quantization;
- evaluates all 462 distinct balanced six-family versus six-family splits;
- samples 300 deterministic disjoint within-family configuration splits (3 versus 3 configurations, seventh left out);
- measures geometry correlations, participation-ratio differences, and five-nearest-task overlap;
- tests greedy 12-task source-to-target correlation-based coverage relative to target-aware greedy and random subsets;
- compares primary reported statistics against `results/reference_20261008.json` within 0.002 numerical tolerance; exits with a nonzero error code for mismatched headline results.

The original baseline verification script and data are preserved. `analysis/reproduce_submission.py` is an additional analysis, not a replacement for the baseline script. A GitHub Actions workflow also runs both checks automatically on the public repository.

## Interpretation and limitations

The analyses reuse the same 72 tasks and the same existing set of 84 configurations, without retraining. They do **not** evaluate entirely new algorithm families, new benchmark suites, actual ranking preservation, causal effects, or deployment outcomes. The distributions from overlapping or complementary partitions are **dependent**, so their percentile intervals are descriptive ranges, not confidence intervals for a superpopulation.

A portfolio nested in the reference has shared measurements with that reference; correlation to it is not an independent transfer validation. We report separate comparisons with disjoint partitions for this reason. The target-aware greedy task subset is an in-sample comparator, **not** a globally optimal oracle. Random-baseline values depend on the explicitly documented deterministic seed schedule.

The original raw UCI/OpenML data and full model-training cache are not redistributed; this repository provides the analysis-ready score matrices. The original results beyond the public analysis layer remain recorded in `verified_reported_results.json`. Reproducibility of the analysis **does not** imply end-to-end reproduction of every model-training experiment.

## Main paths

- `analysis/reproduce_results.py` : prior spectral and portfolio summary checks
- `analysis/reproduce_submission.py` : disjoint geometry, task-neighbor and task-subset transfer analyses
- `results/reference_20261008.json` : published numerical headline checks
- `protocol/FROZEN_PROTOCOL_CC18_K84.json` : originally frozen CC18 K=84 design
- `data/encoded/` : analysis-ready score matrices
- `data/openml_cc18_portfolio_k84.csv` : portfolio mapping

The supplementary ZIP previously distributed for draft preparation has been replaced by these directly accessible public repository materials. Authors must still verify policy compliance, affiliations and submission metadata independently.
