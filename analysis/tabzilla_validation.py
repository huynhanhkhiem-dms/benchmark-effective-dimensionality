#!/usr/bin/env python3
"""Replicate decision-reliability audit on independent PUBLISHED TabZilla score source.

TabZilla scores are taken from naszilla/tabzilla/results_viz/tabzilla-browser.html.
This independent score compilation is NOT guaranteed to have task-disjoint IDs
from CC18. No new model training, new-algorithm families or causal claims.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from rank_transfer_audit import Destination, summarize, spearman_vectors

ROOT=Path(__file__).resolve().parent.parent
SEED=20261008
KS=(6,12,24)
N_SPLITS=120
RANDOM_REPS=12

def similarity(x):
    with np.errstate(divide="ignore",invalid="ignore"):
        r=np.corrcoef(x)
    if not np.isfinite(r).all():
        # Do not allow an undefined correlation to induce hidden exclusion.
        r=np.nan_to_num(r,nan=0,posinf=0,neginf=0)
    np.fill_diagonal(r,1)
    return r

def coverage_order(r,k,how):
    n=len(r)
    if how=="farthest_first":
        start=int(np.argmax(r.mean(axis=0)))
        chosen=[start]
        near=r[:,start].copy()
        while len(chosen)<k:
            # With similarity as affinity, seek a maximally novel task.
            distance=1-near
            distance[chosen]=-np.inf
            j=int(np.argmax(distance))
            chosen.append(j)
            near=np.maximum(near,r[:,j])
        return np.array(chosen)
    if how=="greedy":
        near=np.full(n,-1.0)
        chosen=[]
        for _ in range(k):
            utility=np.maximum(near[:,None],r).sum(axis=0)
            utility[chosen]=-np.inf
            j=int(np.argmax(utility));chosen.append(j)
            near=np.maximum(near,r[:,j])
        return np.array(chosen)
    raise KeyError(how)

def robust_order(x,k):
    # A transparent candidate method, NOT claimed as a novel estimator.
    # Use two disjoint halves within SOURCE to reward consensus of task coverage.
    n=len(x); cut=x.shape[1]//2
    if cut<2 or x.shape[1]-cut<2:
        return coverage_order(similarity(x),k,"greedy")
    r1=similarity(x[:,:cut]);r2=similarity(x[:,cut:])
    chosen=[]; near1=np.full(n,-1.0);near2=np.full(n,-1.0)
    for _ in range(k):
        proposal1=np.maximum(near1[:,None],r1)
        proposal2=np.maximum(near2[:,None],r2)
        utility=np.minimum(proposal1,proposal2).sum(axis=0)
        utility[chosen]=-np.inf
        j=int(np.argmax(utility));chosen.append(j)
        near1=np.maximum(near1,r1[:,j]);near2=np.maximum(near2,r2[:,j])
    return np.array(chosen)

def main():
    score=ROOT/"data"/"tabzilla_independent_published_accuracy.csv"
    if not score.is_file():raise RuntimeError("TabZilla score matrix not sourced")
    table=pd.read_csv(score,index_col=0)
    x=table.to_numpy(dtype=float)
    if not np.isfinite(x).all():raise ValueError("Incomplete matrix")
    if x.shape[0] < 30 or x.shape[1] < 8:raise ValueError("Score matrix too small")
    rng=np.random.default_rng(SEED)
    acc={}
    for split in range(N_SPLITS):
        alg_perm=rng.permutation(x.shape[1])
        a=alg_perm[:x.shape[1]//2];b=alg_perm[x.shape[1]//2:]
        for direction,(source,target) in enumerate(((a,b),(b,a))):
            r=similarity(x[:,source]); dst=Destination(x[:,target])
            source_cols=x[:,source]
            orders={
                "greedy":coverage_order(r,max(KS),"greedy"),
                "farthest_first":coverage_order(r,max(KS),"farthest_first"),
                "source_split_minimax":robust_order(source_cols,max(KS))
            }
            target_order=coverage_order(similarity(x[:,target]),max(KS),"greedy")
            for k in KS:
                seed=SEED+split*1039+direction*983+k*79
                rrng=np.random.default_rng(seed)
                scores={method:dst.assess(order[:k]) for method,order in orders.items()}
                scores["target_aware_greedy"]=dst.assess(target_order[:k])
                scores["random"]=[dst.assess(rrng.choice(len(x),size=k,replace=False))
                                   for _ in range(RANDOM_REPS)]
                if k not in acc:acc[k]={}
                for method in scores:
                    if method not in acc[k]:acc[k][method]={}
                    metric=scores[method]
                    if isinstance(metric,list):
                        metric={key:float(np.mean([row[key] for row in metric])) for key in metric[0]}
                    for key,value in metric.items():
                        acc[k][method].setdefault(key,[]).append(value)
    out={
        "data_source":"Published TabZilla browser scores (2023 NeurIPS study)",
        "data_url":"https://github.com/naszilla/tabzilla/blob/main/results_viz/tabzilla-browser.html",
        "data_matrix_shape":list(x.shape),
        "score_type":"published average test accuracy, task-wise raw score",
        "select_on":"source algorithm subset, no target scores used",
        "partitions":N_SPLITS,
        "oriented_trials":2*N_SPLITS,
        "random_subsets_per_trial":RANDOM_REPS,
        "source_algorithm_split":"random disjoint 50/50 without family-level blocking",
        "methods":{
            "greedy":"source task-correlation coverage",
            "farthest_first":"source correlation farthest-first task diversity",
            "source_split_minimax":"provisional source-half worst-case coverage heuristic; NO novelty claimed",
            "random":"random task selection",
            "target_aware_greedy":"target-informed geometry comparator, not ranking oracle"
        },
        "outcomes":{},
        "limitations":[
            "The TabZilla source may share dataset identifiers with CC18; not independently sampled tasks.",
            "Available complete-case tasks/configurations selected solely by score missingness.",
            "Different algorithm families represented; random halves do not guarantee family-disjoint validation.",
            "Results dependent across repeated partitions of fixed finite matrix; no inferential population confidence intervals.",
            "Candidate minimax heuristic tested post hoc and not claimed as a novel method.",
            "No source-to-target generalization to future algorithms, model fits or deployment outcomes."
        ]}
    for k in KS:
        group={}
        for method in acc[k]:
            group[method]={metric:summarize(values) for metric,values in acc[k][method].items()}
        out["outcomes"][str(k)]=group
    p=ROOT/"results"/"tabzilla_validation_20261008.json"
    p.write_text(json.dumps(out,indent=2,allow_nan=False))
    print("RESULTS WRITTEN:",p)
    for k,v in out["outcomes"].items():
        print(f"Task budget={k}")
        for method,row in v.items():
            print(method,"rho",row["rho_mean"]["median"],"top1",row["top1_same"]["mean"],
                  "regret_x100",row["winner_regret_x100"]["mean"],
                  "eps_.005",row["epsilon_optimal_0_5_x100"]["mean"])
if __name__=="__main__":main()
