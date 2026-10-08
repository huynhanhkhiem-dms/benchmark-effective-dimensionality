#!/usr/bin/env python3
"""Reproduce exploratory CC18 disjoint-portfolio analyses from public score data.

This script does NOT rerun model training; partition quantiles are descriptive,
not confidence intervals. All arms use the same 72 tasks and 84-model repertoire.
"""
import argparse
import base64
import gzip
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261008

def load(repo):
    data = repo / "data"
    enc = data / "encoded"
    meta = json.loads((enc / "openml_cc18_bacc_k84.meta.json").read_text())
    txt = "".join((enc / part).read_text().strip() for part in meta["parts"])
    raw = gzip.decompress(base64.b64decode(txt))
    x = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    x = x.astype(float) / float(meta["scale"])
    portfolio = pd.read_csv(data / "openml_cc18_portfolio_k84.csv")
    assert x.shape == (72, 84) and len(portfolio) == 84
    groups = [np.flatnonzero(portfolio.family.eq(f).to_numpy()).tolist()
              for f in portfolio.family.drop_duplicates()]
    assert len(groups) == 12 and all(len(g) == 7 for g in groups)
    assert np.isfinite(x).all()
    return x, groups

def geometry(x, cols):
    r = np.corrcoef(x[:, cols])
    assert np.isfinite(r).all()
    return r

def correlation(a, b):
    idx = np.triu_indices(72, 1)
    return float(np.corrcoef(a[idx], b[idx])[0, 1])

def participation_ratio(r):
    return float(72**2 / np.square(r).sum())

def neighbors(r):
    z = r.copy()
    np.fill_diagonal(z, -np.inf)
    return np.argsort(-z, axis=1, kind="stable")[:, :5]

def retention(a, b):
    return sum(len(set(x).intersection(y)) for x, y in zip(a, b)) / a.size

def coverage(r, selected):
    return float(r[:, selected].max(axis=1).mean())

def greedy(r):
    near = np.full(72, -1.0)
    selected = []
    for _ in range(12):
        candidate = max((j for j in range(72) if j not in selected),
                        key=lambda j: float(np.maximum(near, r[:, j]).sum()))
        selected.append(candidate)
        near = np.maximum(near, r[:, candidate])
    return selected

def summarize(v):
    z = np.asarray(v, dtype=float)
    q = np.quantile(z, [.025, .5, .975])
    return dict(n=int(len(z)), median=round(float(q[1]), 5),
                q025=round(float(q[0]), 5), q975=round(float(q[2]), 5))

def shuffle(values, state):
    out = list(values)
    for i in range(len(out)-1, 0, -1):
        state = (1664525*state+1013904223) & 0xffffffff
        j = int((state / 4294967296.0) * (i+1))
        out[i], out[j] = out[j], out[i]
    return out, state

def audit(repo):
    x, groups = load(repo)
    reference = geometry(x, list(range(84)))
    reference_near = neighbors(reference)
    output = {"data": "CC18 balanced accuracy, 72 tasks, 84 configurations, quantized 1e-6",
              "seed": SEED, "full_participation_ratio": participation_ratio(reference)}
    for kind in ("family_disjoint", "configuration_disjoint"):
        rhos, a_ref, b_ref, pr_diff = [], [], [], []
        overlap, overlap_ref = [], []
        trans, oracle, random = [], [], []
        state = SEED
        if kind == "family_disjoint":
            parts = []
            for subset in itertools.combinations(range(1, 12), 5):
                lhs = {0, *subset}
                parts.append(([j for i in range(12) if i in lhs for j in groups[i]],
                              [j for i in range(12) if i not in lhs for j in groups[i]]))
        else:
            parts = []
            for _ in range(300):
                a, b = [], []
                for group in groups:
                    z, state = shuffle(group, state)
                    a += z[:3]
                    b += z[3:6]
                parts.append((a, b))
        for ix, (left, right) in enumerate(parts):
            A = geometry(x, left)
            B = geometry(x, right)
            rhos.append(correlation(A, B))
            a_ref.append(correlation(A, reference))
            b_ref.append(correlation(B, reference))
            pr_diff.append(abs(participation_ratio(A) - participation_ratio(B)))
            if kind == "family_disjoint":
                na, nb = neighbors(A), neighbors(B)
                overlap.append(retention(na, nb))
                overlap_ref.append((retention(na, reference_near) +
                                    retention(nb, reference_near))/2)
                if ix % 3 == 0:
                    trans.append(coverage(B, greedy(A)))
                    oracle.append(coverage(B, greedy(B)))
                    score = 0.0
                    for rep in range(12):
                        seed = (SEED + ix*987 + rep*21) & 0xffffffff
                        perm, _ = shuffle(range(72), seed)
                        score += coverage(B, perm[:12])/12
                    random.append(score)
        if kind == "configuration_disjoint":
            # The decision experiment uses a shared LCG stream: random-baseline
            # draws interleave with portfolio draws. Geometry-only draws above
            # deliberately DO NOT interleave those baseline draws.
            state = SEED
            for ix in range(300):
                left, right = [], []
                for group in groups:
                    z, state = shuffle(group, state)
                    left += z[:3]
                    right += z[3:6]
                A = geometry(x, left)
                B = geometry(x, right)
                na, nb = neighbors(A), neighbors(B)
                overlap.append(retention(na, nb))
                overlap_ref.append((retention(na, reference_near) +
                                    retention(nb, reference_near))/2)
                if ix % 3 == 0:
                    trans.append(coverage(B, greedy(A)))
                    oracle.append(coverage(B, greedy(B)))
                    score = 0.0
                    for rep in range(12):
                        perm, state = shuffle(range(72), state)
                        score += coverage(B, perm[:12])/12
                    random.append(score)
        output[kind] = {
            "n": len(parts), "geometry_disjoint": summarize(rhos),
            "geometry_anchor_A": summarize(a_ref),
            "geometry_anchor_B": summarize(b_ref),
            "absolute_PR_difference": summarize(pr_diff),
            "top5_disjoint": summarize(overlap),
            "top5_containing_reference": summarize(overlap_ref),
            "representative_task_selection": {
                "n": len(trans), "subset_size": 12,
                "transfer": summarize(trans), "target_aware_greedy": summarize(oracle),
                "random": summarize(random),
                "mean_transfer_minus_random": float(np.mean(np.array(trans)-np.array(random))),
                "fraction_transfer_below_random": float(np.mean(np.array(trans)<np.array(random)))
            }}
    output["limitations"] = ("Exploratory within-repertoire dependent partitions; quantiles "
                             "are descriptive, not population CIs; coverage is a correlation "
                             "proxy, not preservation of algorithm ranking or performance.")
    return output

def validate(output, reference):
    errors = []
    def ck(name, found, desired):
        if abs(found-desired) > .002:
            errors.append(f"{name}: {found} != {desired}")
    ck("full_PR", output["full_participation_ratio"], reference["full_pr"])
    for label, key in (("family_disjoint", "family"), ("configuration_disjoint", "config")):
        obj = output[label]
        for field, refkey in (("geometry_disjoint", "disjoint_geometry_median"),
                              ("top5_disjoint", "top5_disjoint_median"),
                              ("top5_containing_reference", "top5_reference_median")):
            ck(label + "/" + field, obj[field]["median"], reference[key][refkey])
        task = obj["representative_task_selection"]
        ck(label + "/transfer", task["transfer"]["median"], reference[key]["transfer_median"])
        ck(label + "/random", task["random"]["median"], reference[key]["random_median"])
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path("recomputed_submission_results.json"))
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).resolve().parent.parent /
                        "results" / "reference_20261008.json")
    args = parser.parse_args()
    answer = audit(args.repo)
    args.output.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    if args.reference.is_file():
        issues = validate(answer, json.loads(args.reference.read_text()))
        if issues:
            raise SystemExit("REFERENCE MISMATCH:\n" + "\n".join(issues))
        print("PASS: published headline summaries reproduce within 0.002")
    print(json.dumps(answer, indent=2))
