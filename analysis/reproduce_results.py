#!/usr/bin/env python3
from pathlib import Path
import base64, gzip, io, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENC = DATA / "encoded"

def participation_ratio(df):
    x = np.asarray(df, dtype=float)
    r = np.corrcoef(x)
    eig = np.clip(np.linalg.eigvalsh(r), 0.0, None)
    return float(eig.sum() ** 2 / np.square(eig).sum())

def decode_csv_gz_b64(name):
    raw = gzip.decompress(base64.b64decode((ENC / name).read_text().strip()))
    return pd.read_csv(io.BytesIO(raw), index_col=0)

def decode_cc18():
    meta = json.loads((ENC / "openml_cc18_bacc_k84.meta.json").read_text())
    encoded = "".join((ENC / p).read_text().strip() for p in meta["parts"])
    raw = gzip.decompress(base64.b64decode(encoded))
    x = np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["shape"]).astype(float)
    x /= float(meta["scale"])
    cols = pd.read_csv(DATA / "openml_cc18_portfolio_k84.csv")["config"].tolist()
    return pd.DataFrame(x, columns=cols)

def corr_upper(a, b):
    idx = np.triu_indices_from(a, 1)
    return float(np.corrcoef(a[idx], b[idx])[0, 1])

cls = decode_csv_gz_b64("uci_classification_bacc.csv.gz.b64")
reg = decode_csv_gz_b64("uci_regression_r2c.csv.gz.b64")
cc18 = decode_cc18()

portfolio = pd.read_csv(DATA / "openml_cc18_portfolio_k84.csv")
families = pd.read_csv(DATA / "openml_cc18_family_map.csv")
family_of = dict(zip(families["algo"], families["family"]))

reused = set(portfolio.loc[portfolio["reused_from_layer2_k36"] == True, "config"])
cc36 = cc18[[c for c in cc18.columns if c in reused]]
r36 = np.corrcoef(cc36.to_numpy(float))
r84 = np.corrcoef(cc18.to_numpy(float))

centroids = pd.DataFrame({
    fam: cc18[[c for c in cc18.columns if family_of[c] == fam]].mean(axis=1)
    for fam in sorted(set(family_of.values()))
})
lofo = {}
for fam in sorted(set(family_of.values())):
    cols = [c for c in cc18.columns if family_of[c] != fam]
    lofo[fam] = participation_ratio(cc18[cols])

stability = pd.read_csv(DATA / "openml_cc18_portfolio_stability.csv")
eligible = stability.loc[stability["R_with_K84_q10"] >= 0.90, "K"]
k_star = int(eligible.min())

split = pd.read_csv(DATA / "openml_cc18_split_half_reliability.csv")
reported = json.loads((ROOT / "verified_reported_results.json").read_text())

observed = {
    "uci_classification_pr": participation_ratio(cls),
    "uci_regression_pr": participation_ratio(reg),
    "openml_cc18_pr_quantized": participation_ratio(cc18),
    "cc18_k36_vs_k84_matrix_correlation": corr_upper(r36, r84),
    "cc18_K_star_tau_0.90_q_0.10": k_star,
    "cc18_family_centroid_pr": participation_ratio(centroids),
    "cc18_leave_one_family_out_min": float(min(lofo.values())),
    "cc18_leave_one_family_out_max": float(max(lofo.values())),
    "cc18_split_half_median_spearman_brown": float(split["spearman_brown_reliability"].median()),
}

checks = {
    "uci_classification_pr": abs(observed["uci_classification_pr"] - reported["uci_classification"]["participation_ratio"]) < 1e-10,
    "uci_regression_pr": abs(observed["uci_regression_pr"] - reported["uci_regression"]["participation_ratio"]) < 1e-10,
    "openml_cc18_pr": abs(observed["openml_cc18_pr_quantized"] - reported["openml_cc18"]["participation_ratio"]) < 1e-4,
    "cc18_K_star": observed["cc18_K_star_tau_0.90_q_0.10"] == reported["openml_cc18_portfolio_identification"]["K_star_tau_0.90_q_0.10"],
    "cc18_family_centroid_pr": abs(observed["cc18_family_centroid_pr"] - reported["openml_cc18_portfolio_identification"]["family_centroid"]["participation_ratio"]) < 1e-4,
    "cc18_lofo_range": (
        abs(observed["cc18_leave_one_family_out_min"] - reported["openml_cc18_portfolio_identification"]["leave_one_family_out_min"]) < 1e-4
        and abs(observed["cc18_leave_one_family_out_max"] - reported["openml_cc18_portfolio_identification"]["leave_one_family_out_max"]) < 1e-4
    ),
}

result = {
    "observed_from_public_analysis_ready_matrices": observed,
    "verification_against_full_saved_output_rerun": checks,
    "all_checks_pass": bool(all(checks.values())),
    "note": (
        "The public CC18 matrix is quantized to 1e-6 for compact storage. "
        "Exact manuscript values, permutation summaries, metric-sensitivity results, "
        "held-out subspace-reconstruction null results, and provenance diagnostics from "
        "the full saved-output rerun are retained in verified_reported_results.json."
    ),
}

out = ROOT / "public_reproduction_check.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
