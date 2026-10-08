#!/usr/bin/env python3
"""Pre-submission falsification study for conditional finite-benchmark regret bounds.
The Lipschitz envelope is standard; no algorithmic or theorem novelty is claimed."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from reproduce_submission import load

ROOT=Path(__file__).resolve().parent.parent
SEED=20261008
KS=(6,12,24,36)

def dist(z):
    return np.max(np.abs(z[:,:,None]-z[:,None,:]),axis=0)

def order(d,mode,seed):
    n=len(d)
    if mode=='random':return np.random.default_rng(seed).permutation(n)
    first=int(np.argmin(d.max(axis=1)))
    chosen=[first];near=d[:,first].copy()
    while len(chosen)<max(KS):
        if mode=='minimax':
            util=near.copy();util[chosen]=-1
        else:
            util=np.maximum(0,near[:,None]-d).sum(axis=0);util[chosen]=-1
        j=int(np.argmax(util));chosen.append(j);near=np.minimum(near,d[:,j])
    return np.array(chosen)

def conditional_bound(t,d,s,cand):
    g=t[s,:]-t[s,cand][:,None]
    upper=np.minimum(1.,np.min(g[:,None,:]+2*d[s,:,None],axis=0))
    return float(max(0,np.max(upper.mean(axis=0))))

def experiment():
    x,groups=load(ROOT)
    rng=np.random.default_rng(SEED)
    splits=[]
    for _ in range(48):
        perm=rng.permutation(len(groups))
        a=[j for i in perm[:6] for j in groups[i]]
        b=[j for i in perm[6:] for j in groups[i]]
        splits.extend([(a,b),(b,a)])
    arms={mode:{str(k):{'cond':[],'uncond':[],'regret':[],'cert':[],'cert_verified':[],'false_cert':[]}
          for k in KS} for mode in ('random','minimax','mean_cover')}
    assumptions=[];excesses=[]
    for idx,(src,dest) in enumerate(splits):
        target=x[:,dest]
        d=dist(x[:,src].T)
        excess=np.maximum(0,dist(target.T)-d)
        valid=bool(excess.max()<=1e-12)
        assumptions.append(valid);excesses.append(float(excess.max()))
        for mode in arms:
            selected_order=order(d,mode,SEED+idx*127+len(mode))
            for k in KS:
                s=selected_order[:k]
                cand=int(np.argmax(target[s,:].mean(axis=0)))
                actual=float(target.mean(axis=0).max()-target[:,cand].mean())
                cb=conditional_bound(target,d,s,cand)
                g=target[s,:]-target[s,cand][:,None]
                ub=float(max(0,np.max((g.sum(axis=0)+len(x)-k)/len(x))))
                a=arms[mode][str(k)]
                a['cond'].append(cb);a['uncond'].append(ub);a['regret'].append(actual)
                a['cert'].append(cb<=.01)
                a['cert_verified'].append((cb<=.01) and valid)
                a['false_cert'].append((cb<=.01) and actual>.01)
    report={
       'data':'Public CC18 72-by-84 score matrix, no retraining',
       'oriented_trials':len(splits),
       'conditional_assumption':'Every target column is 1-Lipschitz under source maximum-score difference pseudometric',
       'assumption_valid_fraction':float(np.mean(assumptions)),
       'max_assumption_excess':float(max(excesses)),
       'median_assumption_excess':float(np.median(excesses)),
       'results':{},'novelty_status':'Classical conditional extension; not a new theorem'}
    for name,aa in arms.items():
        report['results'][name]={}
        for k,a in aa.items():
            report['results'][name][k]={
              'conditional_bound_median':float(np.median(a['cond'])),
              'unconditional_bound_median':float(np.median(a['uncond'])),
              'regret_mean':float(np.mean(a['regret'])),
              'conditional_cert_rate_if_assumed':float(np.mean(a['cert'])),
              'verified_assumption_and_cert_rate':float(np.mean(a['cert_verified'])),
              'false_cert_rate_if_assumption_ignored':float(np.mean(a['false_cert']))}
    path=ROOT/'results'/'pcare_certificate_falsification_20261008.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':experiment()
