#!/usr/bin/env python3
"""Q4 rescue audit: finite-task partial-benchmark certificates across fixed public matrices.
EXPLORATORY post-hoc falsification only; 0 inferential claims about populations.
Source and target configurations are always disjoint. No model retraining.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from numpy.random import default_rng
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"analysis"))
from reproduce_submission import load as load_cc18
from rank_transfer_audit import load_uci

def metric(X):
    return np.max(np.abs(X[:,:,None]-X[:,None,:]),axis=0)

def selection(d,k,seed,arm):
    n=len(d)
    if arm=="random": return np.random.default_rng(seed).permutation(n)[:k]
    near=np.full(n,np.inf)
    chosen=[]
    for j in range(k):
        if not chosen:
            idx=int(np.argmin(d.mean(axis=0)))
        elif arm=="max_distance":
            scores=near.copy();scores[chosen]=-np.inf
            idx=int(np.argmax(scores))
        else:
            scores=np.maximum(near[:,None]-d,0).sum(axis=0)
            scores[chosen]=-np.inf
            idx=int(np.argmax(scores))
        chosen.append(idx);near=np.minimum(near,d[:,idx])
    return np.array(chosen,dtype=int)

def eval_one(x,source,target,k,seed,arm):
    z=x[:,source];y=x[:,target];n,m=y.shape
    ds=metric(z.T)
    dt=metric(y.T)
    nz=ds>1e-10
    zero_conflict=(~nz)&(dt>1e-8)
    violations=dt>ds+1e-10
    min_scale=float(np.max(dt[nz]/ds[nz])) if nz.any() else None
    if zero_conflict.any(): min_scale=None
    # Pairwise Lipschitz check used directly by regret certificate:
    # Is every target gap 2-Lipschitz under the source pseudometric?
    # Exhaustive directed pairwise gaps evaluated for all target configurations.
    # The sufficient column condition is not conflated with this weaker pairwise condition.
    gapfail=False
    for a in range(m):
        for b in range(a+1,m):
            vals=y[:,b]-y[:,a]
            if (np.abs(vals[:,None]-vals[None,:])>2*ds+1e-10).any():
                gapfail=True;break
        if gapfail:break
    s=selection(ds,k,seed,arm)
    avg=y.mean(axis=0);candidate=int(np.argmax(y[s].mean(axis=0)))
    regret=float(np.max(avg)-avg[candidate])
    others=np.arange(m)!=candidate
    g=y[s,:]-y[s,candidate][:,None]
    env=np.minimum(1,np.min(g[:,None,:]+2*ds[s,:,None],axis=0))
    cond=float(max(0, np.max(env[:,others].mean(axis=0))))
    uncond=float(max(0,np.max((g[:,others].sum(axis=0)+(n-k))/n)))
    # An all-task "oracle" metric multiplier only for sensitivity, NEVER a deployable certificate.
    if min_scale is not None and np.isfinite(min_scale):
        lam=max(1.,min_scale)
        oracle_env=np.minimum(1,np.min(g[:,None,:]+2*lam*ds[s,:,None],axis=0))
        retrospective=float(max(0,np.max(oracle_env[:,others].mean(axis=0))))
    else:
        retrospective=None
    return dict(regret=regret,invalid_column=bool(violations.any()),
      invalid_gap=gapfail, minimal_column_scale=min_scale,
      taskpair_violation_fraction=float(np.mean(violations[np.triu_indices(n,1)])),
      zero_metric_conflict=bool(zero_conflict.any()),
      conditional_bound=cond,unconditional_bound=uncond,
      conditional_appears_certified=bool(cond<=.01),
      retrospective_bound=retrospective)

def splits_cc18(groups,rng,count=48):
    for _ in range(count):
        perm=rng.permutation(len(groups))
        a=[j for i in perm[:6] for j in groups[i]]
        b=[j for i in perm[6:] for j in groups[i]]
        yield a,b;yield b,a

def splits_two(ncols,rng,count=48):
    for _ in range(count):
        perm=rng.permutation(ncols)
        n=ncols//2
        yield perm[:n].tolist(),perm[n:].tolist()
        yield perm[n:].tolist(),perm[:n].tolist()

def splits_tabzilla(cols,rng,count=48):
    families={
      "gbdt": {"CatBoost","LightGBM","XGBoost"},
      "baselines": {"DecisionTree","KNN","LinearModel","RandomForest","SVM"}
    }
    families["neural"]=set(cols)-families["gbdt"]-families["baselines"]
    assert all(families.values()) and len(set.union(*families.values()))==len(cols)
    # Disjoint model families, no partition that splits a family.
    # Select one family as target, two as source, repeated only 3 orientations.
    # Repetition is avoided so each comparison remains a distinct study arm.
    for targetfam in ("gbdt","baselines","neural"):
        target=[i for i,n in enumerate(cols) if n in families[targetfam]]
        src=[i for i,n in enumerate(cols) if n not in families[targetfam]]
        yield src,target

def summarize(vals):
    x=np.asarray([v for v in vals if v is not None],dtype=float)
    if not len(x):return None
    return {"n":int(len(x)),"mean":float(x.mean()),"median":float(np.median(x)),
     "min":float(x.min()),"max":float(x.max()),"q025_descriptive":float(np.quantile(x,.025)),
     "q975_descriptive":float(np.quantile(x,.975))}

def audit(name,x,parts,klist=(6,12,24),):
    outcomes=[]
    for i,(src,dst) in enumerate(parts):
        for k in klist:
            if k>=x.shape[0]: continue
            for arm in ("random","max_distance","mean_cover"):
                o=eval_one(x,src,dst,k,20261008+i*97+k*13+len(arm),arm)
                o.update(partition=i,k=k,arm=arm)
                outcomes.append(o)
    summary={"name":name,"shape":list(x.shape),"distinct_partitions":len({v["partition"] for v in outcomes}),
      "partition_quantiles_are_descriptive":True,"cells":{}}
    for k in klist:
      if k>=len(x):continue
      for arm in ("random","max_distance","mean_cover"):
        subset=[o for o in outcomes if o["k"]==k and o["arm"]==arm]
        if not subset:continue
        prefix=f"k{k}_{arm}"
        summary["cells"][prefix]={
          "n":len(subset),
          "column_1l_violation":sum(o["invalid_column"] for o in subset),
          "pairwise_gap_2l_violation":sum(o["invalid_gap"] for o in subset),
          "source_zero_metric_conflict":sum(o["zero_metric_conflict"] for o in subset),
          "rate_conditional_claim_at_epsilon_001":float(np.mean([o["conditional_appears_certified"] for o in subset])),
          "regret":summarize([o["regret"] for o in subset]),
          "conditional_envelope_invalid_if_gap_violation":summarize([o["conditional_bound"] for o in subset]),
          "sharp_unconditional_bound":summarize([o["unconditional_bound"] for o in subset]),
          "retrospective_oracle_bound_not_a_certificate":summarize([o["retrospective_bound"] for o in subset]),
          "min_column_lipschitz_scale":summarize([o["minimal_column_scale"] for o in subset]),
          "taskpair_column_violation_fraction":summarize([o["taskpair_violation_fraction"] for o in subset])
        }
    return summary

def main():
    x,groups=load_cc18(ROOT)
    tab=pd.read_csv(ROOT/"data"/"tabzilla_independent_published_accuracy.csv",index_col=0)
    arrays={"cc18_6family_vs_6family":audit("CC18 disjoint six-family partitions",x,
             splits_cc18(groups,default_rng(20261008))),
      "tabzilla_leave_one_family_out":audit("TabZilla source-score families with one family held out",
             tab.to_numpy(float),splits_tabzilla(tab.columns.tolist(),default_rng(20261008))),
      "uci_classification_random_disjoint":audit("UCI classification disjoint configurations",
             load_uci("uci_classification_bacc.csv.gz.b64"),
             splits_two(36,default_rng(20261008))),
      "uci_regression_random_disjoint":audit("UCI regression disjoint configurations, clipped R2",
             load_uci("uci_regression_r2c.csv.gz.b64"),
             splits_two(34,default_rng(20261008)))}
    out={"study_design":"Exploratory finite-matrix falsification: 4 nonindependent score sources; no population inference",
         "important_limits":["CC18 and TabZilla may share benchmark tasks; they are separate score compilations, not guaranteed task independent.",
          "UCI classification and regression task sets partly originate from historically related sources.",
          "TabZilla family arms have only 3 distinct partitions; do not extrapolate frequency.",
          "Some score sources use clipped regression R2; regret interpretation remains in source score units.",
          "The retrospective multiplier uses full target scores and is not deployable.",
          "No metric valid without verified target smoothness or independent random audit."],
         "datasets":arrays}
    f=ROOT/"results"/"pcare_crossmatrix_q4_20261008.json"
    f.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    for name,r in arrays.items():
        print(name,"shape",r["shape"],"parts",r["distinct_partitions"])
        for key,v in r["cells"].items():
            if key.startswith("k12_"):
                print(key,"gap-invalid",v["pairwise_gap_2l_violation"],"/",v["n"],
                      "median bound",round(v["conditional_envelope_invalid_if_gap_violation"]["median"],4),
                      "avg regret",round(v["regret"]["mean"],4),
                      "min multiplier",None if v["min_column_lipschitz_scale"] is None else round(v["min_column_lipschitz_scale"]["median"],2))
    print("DONE",f)
if __name__=="__main__":main()
