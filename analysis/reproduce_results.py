#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'data'
OUT = ROOT / 'verified_reported_results.json'


def read_matrix(file):
    return pd.read_csv(D / file, index_col=0)


def pr(M):
    X = M.to_numpy(float) if hasattr(M, 'to_numpy') else np.asarray(M, float)
    R = np.corrcoef(X)
    w = np.clip(np.linalg.eigvalsh(R), 0, None)
    return float((w.sum() ** 2) / (w @ w))


def loo(M, ks=(1, 3, 6, 12)):
    X = M.to_numpy(float)
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, ddof=0, keepdims=True)
    out = {k: [] for k in ks}
    for i in range(len(Z)):
        tr = np.delete(Z, i, 0)
        te = Z[i]
        mu = tr.mean(0)
        _, _, vt = np.linalg.svd(tr - mu, full_matrices=False)
        den = ((te - mu) ** 2).sum()
        for k in ks:
            C = vt[:k]
            rec = mu + (te - mu) @ C.T @ C
            out[k].append(1 - ((te - rec) ** 2).sum() / den)
    return {str(k): float(np.mean(v)) for k, v in out.items()}


def null_values(file):
    return pd.read_csv(D / file).iloc[:, -1].to_numpy(float)


def null_summary(a):
    return {
        'mean': float(a.mean()),
        'q025': float(np.quantile(a, .025)),
        'q975': float(np.quantile(a, .975)),
        'n': int(len(a)),
    }


def mc_lower(vals, obs):
    return float((np.sum(vals <= obs) + 1) / (len(vals) + 1))


def corr_upper(A, B):
    t = np.triu_indices_from(A, 1)
    return float(np.corrcoef(A[t], B[t])[0, 1])


res = {}
settings = [
    ('uci_classification', 'uci_classification_scores_bacc.csv', 'uci_classification_pr_permutation_5000.csv'),
    ('uci_regression', 'uci_regression_scores_r2c.csv', 'uci_regression_pr_permutation_5000.csv'),
    ('openml_cc18', 'openml_cc18_scores_bacc.csv', 'openml_cc18_pr_permutation_5000.csv'),
]
for key, score_file, null_file in settings:
    S = read_matrix(score_file)
    n, k = S.shape
    obs = pr(S)
    nv = null_values(null_file)
    res[key] = {
        'shape': [int(n), int(k)],
        'participation_ratio': obs,
        'permutation': {**null_summary(nv), 'monte_carlo_p_lower': mc_lower(nv, obs)},
        'analytical_null_NK_over_NplusK': float(n * k / (n + k)),
        'loo_r2': loo(S),
    }

cap = read_matrix('uci_classification_cap2000_bacc.csv')
affected = [x.strip() for x in (D / 'uci_classification_cap2000_affected.txt').read_text().splitlines() if x.strip()]
res['uci_classification_cap2000'] = {
    'affected_datasets': len(affected),
    'participation_ratio': pr(cap),
}

S84 = read_matrix('openml_cc18_scores_bacc.csv')
port = pd.read_csv(D / 'openml_cc18_portfolio_k84.csv')
family_map = pd.read_csv(D / 'openml_cc18_family_map.csv').set_index('algo')['family'].to_dict()
old = set(port.loc[port.reused_from_layer2_k36 == True, 'config'])
S36 = S84[[a for a in S84.columns if a in old]]
R84 = np.corrcoef(S84.to_numpy(float))
R36 = np.corrcoef(S36.to_numpy(float))
stab = pd.read_csv(D / 'openml_cc18_portfolio_stability.csv')
split = pd.read_csv(D / 'openml_cc18_split_half_reliability.csv')
crit = stab[stab.R_with_K84_q10 >= .90].sort_values('K').iloc[0]

families = sorted(set(family_map.values()))
cent = pd.DataFrame(index=S84.index)
for f in families:
    cols = [a for a in S84.columns if family_map[a] == f]
    cent[f] = S84[cols].mean(axis=1)
cent_pr = pr(cent)
rng = np.random.default_rng(2026)
X = cent.to_numpy(float)
cn = np.empty(5000)
for i in range(5000):
    cn[i] = pr(np.vstack([r[rng.permutation(X.shape[1])] for r in X]))

lo = {}
for f in families:
    cols = [a for a in S84.columns if family_map[a] != f]
    lo[f] = pr(S84[cols])

res['openml_cc18_portfolio_identification'] = {
    'original_K36_vs_K84_matrix_correlation': corr_upper(R36, R84),
    'stability_curve': {
        str(int(r.K)): {
            'median': float(r.R_with_K84_median),
            'q10': float(r.R_with_K84_q10),
            'q90': float(r.R_with_K84_q90),
        }
        for r in stab.itertuples(index=False)
    },
    'K_star_tau_0.90_q_0.10': int(crit.K),
    'split_half': {
        'median_matrix_correlation': float(split.between_half_R_matrix_correlation.median()),
        'median_spearman_brown': float(split.spearman_brown_reliability.median()),
        'q025_spearman_brown': float(split.spearman_brown_reliability.quantile(.025)),
        'q975_spearman_brown': float(split.spearman_brown_reliability.quantile(.975)),
    },
    'family_centroid': {
        'participation_ratio': cent_pr,
        'permutation_mean': float(cn.mean()),
        'permutation_q025': float(np.quantile(cn, .025)),
        'permutation_q975': float(np.quantile(cn, .975)),
        'monte_carlo_p_lower': mc_lower(cn, cent_pr),
        'seed': 2026,
        'replicates': 5000,
    },
    'leave_one_family_out': lo,
    'leave_one_family_out_min': float(min(lo.values())),
    'leave_one_family_out_max': float(max(lo.values())),
}

res['metric_sensitivity'] = {
    'uci_classification': {
        m: pr(read_matrix(f'uci_classification_scores_{m}.csv')) for m in ['acc', 'bacc', 'f1m']
    },
    'uci_regression': {
        m: pr(read_matrix(f'uci_regression_scores_{m}.csv')) for m in ['r2c', 'nrmse']
    },
    'openml_cc18': {
        m: pr(read_matrix(f'openml_cc18_scores_{m}.csv')) for m in ['acc', 'bacc', 'f1m']
    },
}

ln = pd.read_csv(D / 'openml_cc18_loo_null_bacc_300.csv')
for k, obs in list(res['openml_cc18']['loo_r2'].items()):
    v = ln[f'k{k}'].to_numpy(float)
    res['openml_cc18']['loo_r2'][k] = {
        'observed': obs,
        'null_mean': float(v.mean()),
        'null_q025': float(np.quantile(v, .025)),
        'null_q975': float(np.quantile(v, .975)),
        'monte_carlo_p_upper': float((np.sum(v >= obs) + 1) / (len(v) + 1)),
    }

links = pd.read_csv(D / 'uci_classification_linked_pairs.csv')
S = read_matrix('uci_classification_scores_bacc.csv')
R = np.corrcoef(S.to_numpy(float))
names = list(S.index)
linkset = {tuple(sorted((r.a, r.b))) for r in links.itertuples(index=False)}
linked, other = [], []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pair = tuple(sorted((names[i], names[j])))
        rho = float(R[i, j])
        (linked if pair in linkset else other).append(rho)
pa = {'linked_mean_rho': float(np.mean(linked)), 'other_mean_rho': float(np.mean(other))}
for t in [.6, .7, .8, .9]:
    strong = [x for x in linked + other if x > t]
    ls = [x for x in linked if x > t]
    pa[f'rho_gt_{t}'] = {
        'strong_pairs': len(strong),
        'linked_pairs': len(ls),
        'unlinked_share': float(1 - len(ls) / len(strong)) if strong else None,
    }
res['uci_classification_provenance_response_audit'] = pa

OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
print(json.dumps(res, indent=2))
