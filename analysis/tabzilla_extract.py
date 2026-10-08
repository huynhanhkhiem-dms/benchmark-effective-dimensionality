#!/usr/bin/env python3
"""Source-verified extraction probe, TabZilla Bokeh published results (no fabricated data)."""
import urllib.request,re,json,base64,html
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parent.parent
URL="https://raw.githubusercontent.com/naszilla/tabzilla/main/results_viz/tabzilla-browser.html"
REPORT=ROOT/"results"/"tabzilla_extraction_diagnostics.json"
report={"source":URL,"status":"PROBE"}
try:
    with urllib.request.urlopen(urllib.request.Request(URL,headers={"User-Agent":"Academic-replication/1.0"}),timeout=55) as res:
        page=res.read().decode("utf-8","replace")
    report["page_size"]=len(page)
    report["head_excerpt"]=page[:250]
    report["term_positions"]={p:page.find(p) for p in ["const docs_json =","'ColumnDataSource'","\"ColumnDataSource\"","alg_name","Accuracy__test_mean","Bokeh.embed.embed_items"]}
    match=re.search(r"const docs_json = document.getElementById\\('([^']+)'\\).textContent",page)
    if not match: raise ValueError("Bokeh JSON script ID not located")
    docid=match.group(1)
    matchdoc=re.search(r'<script[^>]*id="'+re.escape(docid)+r'"[^>]*>(.*?)</script>',page,re.S)
    if not matchdoc:raise ValueError("Embedded script for "+docid+" missing")
    raw=matchdoc.group(1).strip()
    doc=json.loads(raw)
    n=len(raw)
    report["data_script_id"]=docid
    report["doc_keys"]=list(doc)[:15]
    report["bokeh_head_context"]=page[2500:4200]
    report["bokeh_alg_context"]=page[37500:39700]
    report["bokeh_tail_context"]=page[5381500:5382900]
    report["doc_type"]=type(doc).__name__
    report["doc_json_chars"]=n
    candidates=[]
    def decode(value):
        if isinstance(value,list): return [decode(v) for v in value]
        if isinstance(value,dict):
            if "__ndarray__" in value:
                raw=base64.b64decode(value["__ndarray__"])
                return np.frombuffer(raw, dtype=np.dtype(value.get("dtype","float64"))).tolist()
            if value.get("type")=="ndarray" and "array" in value:
                if isinstance(value["array"],list): return value["array"]
                packed=value["array"]
                data=base64.b64decode(packed.get("data","") if isinstance(packed,dict) else packed)
                dtype=np.dtype(value.get("dtype","float64"))
                shape=value.get("shape")
                x=np.frombuffer(data,dtype=dtype)
                if shape: x=x.reshape(shape)
                return x.tolist()
            if value.get("type")=="map" and "entries" in value:
                return {k:decode(v) for k,v in value["entries"]}
            return {str(k):decode(v) for k,v in value.items()}
        return value
    def walk(value,path=""):
        if isinstance(value,dict):
            # Match 2.x model or 3.x schema data
            for cand in (value, value.get("attributes"),value.get("data")):
                if isinstance(cand,dict):
                    target=cand.get("data",cand)
                    target=decode(target)
                    if isinstance(target,dict) and all(k in target for k in ("alg_name","dataset_name")):
                        candidates.append((path,target))
            for k,v in value.items():walk(v,path+"/"+str(k))
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,path+"/"+str(i))
    walk(doc)
    report["candidate_count"]=len(candidates)
    report["candidate_sizes"]=[{"path":path,"length":len(d.get("alg_name",[])),"keys":list(d)[:40]} for path,d in candidates[:8]]
    if not candidates: raise ValueError("No published Bokeh score table found")
    path, data=max(candidates,key=lambda t:len(t[1].get("alg_name",[])))
    allcols={k:v for k,v in data.items() if isinstance(v,list) and len(v)==len(data["alg_name"])}
    report["selected_path"]=path
    report["selected_columns"]=list(allcols)
    df=pd.DataFrame(allcols)
    report["rows"]=len(df)
    report["distinct_algorithms"]=int(df.alg_name.nunique())
    report["distinct_tasks"]=int(df.dataset_name.nunique())
    numeric=[k for k in ("Accuracy","Accuracy (Normalized 0-1)","F1","AUC","Log Loss") if k in df]
    report["numeric_candidates"]=numeric
    if not numeric:raise ValueError("No declared accuracy metric in browser")
    metric="Accuracy" if "Accuracy" in df else numeric[0]
    df[metric]=pd.to_numeric(df[metric],errors="coerce")
    df=df.dropna(subset=[metric])
    wide=df.pivot_table(index="dataset_name",columns="alg_name",values=metric,aggfunc="mean")
    report["pivot_shape"]=[int(v) for v in wide.shape]
    report["pivot_missing_fraction"]=float(wide.isna().mean().mean())
    # retain tasks and models with >= 80% coverage via iterative filtering (not cherry-picking on outcomes).
    rows, cols=wide.copy(),wide.copy()
    mat=wide.copy()
    for _ in range(6):
        mat=mat.loc[mat.notna().sum(axis=1)>=max(3,int(np.ceil(.8*mat.shape[1])))]
        mat=mat.loc[:,mat.notna().sum(axis=0)>=max(3,int(np.ceil(.8*mat.shape[0])))]
    mat=mat.dropna(axis=0).dropna(axis=1)
    report["complete_shape"]=[int(v) for v in mat.shape]
    if min(mat.shape)<5: raise ValueError("Insufficient complete TabZilla score matrix, keep diagnostic only")
    target=ROOT/"data"/"tabzilla_independent_published_accuracy.csv"
    mat.to_csv(target)
    report["dataset_file"]=str(target.relative_to(ROOT))
    report["status"]="SCORE_MATRIX_EXTRACTED"
except Exception as e:
    report["status"]="EXTRACTION_BLOCKED"
    report["error"]=str(e)
REPORT.parent.mkdir(exist_ok=True,parents=True)
REPORT.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
print(json.dumps(report,indent=2,default=str))
