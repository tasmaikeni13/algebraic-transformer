#!/usr/bin/env python3
"""Seed-bootstrap summaries for wider links and paired learned recall.

These descriptive intervals do not correct for searching over links or
establish language-model quality. Random base rows are reused across gains.
"""

import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.attention_statistics import mean_ci


def main():
    paths={"routing":ROOT/"results/analysis/attention_link_wide_analysis.json",
           "recall":ROOT/"results/analysis/attention_recall_probe.json"}
    routing=json.loads(paths["routing"].read_text())
    recall=json.loads(paths["recall"].read_text())
    report={"schema":1,"sources":{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
            "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope":"descriptive independent-seed bootstrap; shared seeds/rows across links and gains; toy recall is not language-model quality; no multiplicity correction",
            "parity_achieved":False,"alternative_model_link_adopted":False,"routing":[],"recall":[]}
    for name in routing["specs"]:
        for gain in routing["gain_products"]:
            rows=sorted((x for x in routing["trials"] if x["kernel"]==name and x["gain_product"]==gain),key=lambda x:x["seed"])
            report["routing"].append({"link":name,"gain_product":gain,"seeds":len(rows),
                "independent_rows":len(rows)*routing["rows_per_seed"],"cue_tie_rows":sum(x["cue_tie_rows"] for x in rows),
                "metrics":{m:mean_ci([x[m] for x in rows]) for m in
                           ("cue_probability_mean","TV_to_deployed_asinh_mean","fraction_scores_clipped","entropy_mean")}})
    for items in recall["items"]:
        reference=sorted((x for x in recall["trials"] if x["link"]=="asinh16_c8" and x["items"]==items),key=lambda x:x["seed"])
        for name in recall["links"]:
            rows=sorted((x for x in recall["trials"] if x["link"]==name and x["items"]==items),key=lambda x:x["seed"])
            if [x["seed"] for x in rows] != [x["seed"] for x in reference]:
                raise ValueError("recall pairs are incomplete")
            report["recall"].append({"items":items,"link":name,"seeds":len(rows),
                "accuracy":mean_ci([x["accuracy"] for x in rows]),"nll":mean_ci([x["nll"] for x in rows]),
                "paired_accuracy_difference_to_asinh":mean_ci([x["accuracy"]-r["accuracy"] for x,r in zip(rows,reference)]),
                "paired_nll_difference_to_asinh":mean_ci([x["nll"]-r["nll"] for x,r in zip(rows,reference)])})
    out=ROOT/"results/analysis/attention_followup_statistics.json"
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(len(report["routing"]),len(report["recall"]))


if __name__=="__main__":
    main()
