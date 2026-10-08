#!/usr/bin/env python3
"""Exploratory cross-portfolio rank transfer, using public analysis-ready matrices.

Study constraints:
- Source task selection sees SOURCE configurations only; destination results are read
  for evaluation, not selection (except explicitly labeled target-aware comparator).
- Random baselines use independent deterministic seeds per split, arm, and k.
- Results share tasks/configurations across overlapping partitions; partition
  quantiles are descriptive, NOT population confidence intervals or p-values.
- This is ranking transfer within existing observed configurations, NOT accuracy
  on future models, unseen families, or fully independent benchmark collections.
"""
import argparse
import base64
import gzip
import io
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from reproduce_submission import load as load_cc18

SEED = 20261008
KS = (6, 12, 24)
RANDOM_REPS = 16
SPLITS_RANDOM = 300


def load_uci(name):
    encoded = (ROOT / "data" / "encoded" / name).read_text().strip()
    frame = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(encoded))),
                        index_col=0)
    x = frame.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains nonfinite entries")
    return x


def make_geometry(x):
    r = np.corrcoef(x)
    if not np.isfinite(r).all():
        raise ValueError("Undefined task correlation")
    return r


def greedy_order(r, count):
    n = len(r)
    best = np.full(n, -1.0)
    out = []
    for _ in range(count):
        utilities = np.maximum(best[:, None], r).sum(axis=0)
        if out:
            utilities[out] = -np.inf
        j = int(np.argmax(utilities))
        out.append(j)
        best = np.maximum(best, r[:, j])
    return np.asarray(out, dtype=int)


def spearman_vectors(a, b):
    ar = rankdata(a, method="average")
    br = rankdata(b, method="average")
    a0, b0 = ar-ar.mean(), br-br.mean()
    den = np.sqrt((a0*a0).sum() * (b0*b0).sum())
    return float((a0 @ b0) / den) if den > 0 else float("nan")


class Destination:
    def __init__(self, values):
        self.x = values
        self.full_score = values.mean(axis=0)
        self.full_borda = -rankdata(-values, axis=1, method="average").mean(axis=0)
        self.per_task_rank = -rankdata(-values, axis=1, method="average")
        self.winner = int(np.argmax(self.full_score))

    def assess(self, subset):
        mean = self.x[subset, :].mean(axis=0)
        borda = self.per_task_rank[subset, :].mean(axis=0)
        selected_winner = int(np.argmax(mean))
        regret = 100*float(
            self.full_score[self.winner]-self.full_score[selected_winner])
        return {
            "rho_mean": spearman_vectors(mean, self.full_score),
            "rho_borda": spearman_vectors(borda, self.full_borda),
            "top1_same": float(selected_winner == self.winner),
            "winner_regret_x100": regret,
            "epsilon_optimal_0_1_x100": float(regret <= 0.1+1e-12),
            "epsilon_optimal_0_5_x100": float(regret <= 0.5+1e-12),
            "epsilon_optimal_1_0_x100": float(regret <= 1.0+1e-12)
        }


def summarize(a):
    arr = np.asarray(a, dtype=float)
    good = arr[np.isfinite(arr)]
    if len(good) == 0:
        return {"n": 0, "median": None}
    return {
        "n": len(good),
        "median": round(float(np.median(good)), 5),
        "q025_descriptive": round(float(np.quantile(good, .025)), 5),
        "q975_descriptive": round(float(np.quantile(good, .975)), 5),
        "mean": round(float(np.mean(good)), 5)
    }


def family_splits(groups):
    # 462 unique unordered partitions: fixing family index 0 on one side.
    for comb in itertools.combinations(range(1, 12), 5):
        left_groups = {0, *comb}
        left = [j for i in range(12) if i in left_groups for j in groups[i]]
        right = [j for i in range(12) if i not in left_groups for j in groups[i]]
        yield left, right


def same_family_configuration_splits(groups):
    rng = np.random.default_rng(SEED)
    for _ in range(SPLITS_RANDOM):
        left, right = [], []
        for group in groups:
            perm = rng.permutation(group)
            left.extend(map(int,perm[:3]))
            right.extend(map(int,perm[3:6]))
        yield left, right


def unstructured_configuration_splits(x, offset):
    rng = np.random.default_rng(SEED+offset)
    m = x.shape[1]
    for _ in range(SPLITS_RANDOM):
        perm = rng.permutation(m)
        p = m // 2
        yield perm[:p].tolist(), perm[p:2*p].tolist()


def audit(x, partitions, design):
    accumulator = defaultdict(list)
    n_pairs = 0
    example_n = 0
    for ix, (left, right) in enumerate(partitions):
        n_pairs += 1
        for orient, (source, target) in enumerate(((left, right),(right, left))):
            source_geom = make_geometry(x[:, source])
            target_values = x[:, target]
            destination = Destination(target_values)
            task_order = greedy_order(source_geom, max(KS))
            target_order = greedy_order(make_geometry(target_values), max(KS))
            for k in KS:
                source_res = destination.assess(task_order[:k])
                oracle_res = destination.assess(target_order[:k])
                seed = (SEED+971*ix+104729*orient+131*k+31*len(design)) & 0xffffffff
                rng = np.random.default_rng(seed)
                random_res = [destination.assess(rng.choice(len(x), size=k, replace=False))
                              for _ in range(RANDOM_REPS)]
                name = f"k{k}"
                for key in source_res:
                    rand = float(np.mean([d[key] for d in random_res]))
                    accumulator[(name,key,"source")].append(source_res[key])
                    accumulator[(name,key,"random")].append(rand)
                    accumulator[(name,key,"target_aware")].append(oracle_res[key])
                    accumulator[(name,key,"source_minus_random")].append(source_res[key]-rand)
                    accumulator[(name,key,"source_minus_oracle")].append(
                        source_res[key]-oracle_res[key])
                if k == 12 and len(accumulator[(name,"rho_mean","source")]) <= 5:
                    example_n += 1
    result = {
        "design": design,
        "tasks": int(x.shape[0]), "configurations": int(x.shape[1]),
        "unique_unordered_partitions": n_pairs,
        "oriented_trials": n_pairs*2,
        "k_values": list(KS), "random_subsets_per_oriented_trial": RANDOM_REPS,
        "descriptive_only": True,
        "metrics": {}
    }
    for k in KS:
        group = f"k{k}"
        obj = {}
        for key in ("rho_mean","rho_borda","top1_same","winner_regret_x100","epsilon_optimal_0_1_x100","epsilon_optimal_0_5_x100","epsilon_optimal_1_0_x100"):
            obj[key] = {
                kind: summarize(accumulator[(group,key,kind)])
                for kind in ("source","random","target_aware","source_minus_random",
                             "source_minus_oracle")}
            diff = np.asarray(accumulator[(group,key,"source_minus_random")])
            if key in ("rho_mean", "rho_borda", "top1_same", "epsilon_optimal_0_1_x100", "epsilon_optimal_0_5_x100", "epsilon_optimal_1_0_x100"):
                obj[key]["fraction_source_worse_than_random_mean"] = round(
                    float(np.mean(diff < 0)),5)
            else:
                obj[key]["fraction_source_worse_than_random_mean"] = round(
                    float(np.mean(diff > 0)),5)
        result["metrics"][group] = obj
    return result


def main():
    x_cc18, groups = load_cc18(ROOT)
    cls = load_uci("uci_classification_bacc.csv.gz.b64")
    reg = load_uci("uci_regression_r2c.csv.gz.b64")
    answer = {
        "status": "exploratory_rank_transfer_on_existing_matrices",
        "protocol_date": "2026-10-08",
        "designs": {
            "cc18_family_disjoint": audit(x_cc18, family_splits(groups),
                                         "CC18 6-versus-6 disjoint families"),
            "cc18_config_disjoint": audit(x_cc18,
                                        same_family_configuration_splits(groups),
                                        "CC18 3-versus-3 within each family, one config omitted"),
            "uci_classification_config_disjoint": audit(cls,
                                        unstructured_configuration_splits(cls, 41),
                                        "UCI classification random half configurations"),
            "uci_regression_config_disjoint": audit(reg,
                                        unstructured_configuration_splits(reg, 73),
                                        "UCI regression random half configurations")
        },
        "limits": [
            "All partitions in a design share tasks and configurations; descriptive quantiles only.",
            "Source-selected tasks are evaluated on an observed complementary target portfolio.",
            "Target-aware comparator has access to target geometry; it is NOT an optimized oracle for ranking preservation.",
            "UCI splits do not preserve family structure: not a replication of CC18 family-disjoint design.",
            "UCI regression results use previously clipped R-squared scores, not balanced accuracy.",
            "Winner regret is 100 times the raw score gap, i.e. percentage points ONLY for balanced accuracy, not for clipped R-squared.",
            "Epsilon-optimal means selected winner has full-target score regret <=0.001, 0.005 or 0.01 in native score units; not a statistical equivalence test.",
            "No evaluation on unseen families, external trials or unseen data.",
            "Not directly comparable to selection benchmarks based on different score-normalization conventions.",
            "Existing analysis-ready data are used; model-training caches were not rerun."
        ]
    }
    output = ROOT / "results" / "rank_transfer_audit_20261008.json"
    output.write_text(json.dumps(answer, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    print("RANK TRANSFER AUDIT COMPLETED")
    for d, v in answer["designs"].items():
        print(d, "n", v["oriented_trials"])
        for k in KS:
            m=v["metrics"][f"k{k}"]
            print(f"  k={k}: rho(source/random) "
                  f"{m['rho_mean']['source']['median']}/"
                  f"{m['rho_mean']['random']['median']}; "
                  f"top1(source/random) "
                  f"{m['top1_same']['source']['mean']}/"
                  f"{m['top1_same']['random']['mean']}; "
                  f"regret pp(source/random) "
                  f"{m['winner_regret_x100']['source']['mean']}/"
                  f"{m['winner_regret_x100']['random']['mean']}")
    print("SAVED:", output)


if __name__ == "__main__":
    main()
