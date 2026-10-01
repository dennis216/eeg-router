"""Post-hoc scorer robustness: cardinality-first matching, no retuning."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path("/root/b_false_alarm_atlas")
SOURCE = ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1"
ORACLE = ROOT / "outputs/reports/h4-protocol-matched-oracle-v2"
AUDIT = ROOT / "outputs/reports/h4-final-linkage-expert-audit-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(module); return module


def cardinality_match(predictions, truths):
    if not predictions or not truths: return []
    overlap = np.zeros((len(predictions), len(truths)))
    for i, p in enumerate(predictions):
        for j, (a, b) in enumerate(truths): overlap[i, j] = max(0.0, min(p[1], b) - max(p[0], a))
    cost = np.where(overlap >= 1.0, -1e6 - overlap, 0.0)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if overlap[i, j] >= 1.0]


def score(events, starts, patients, seizures, durations, h4):
    tp = fp = truth_n = 0; hours = 0.0
    for recording in patients:
        rec = events[events.recording.astype(str) == recording] if not events.empty else events
        pred = [(float(x.event_start), float(x.event_end), float(x.peak_probability)) for x in rec.sort_values("event_start").itertuples(index=False)] if not rec.empty else []
        truth = h4._truths(seizures, recording, starts[recording])
        pairs = cardinality_match(pred, truth)
        tp += len(pairs); fp += len(pred)-len({i for i,_ in pairs}); truth_n += len(truth)
        hours += max(0.0, durations[recording]-starts[recording]-sum(max(0.0,b-a) for a,b in truth))/3600
    return {"tp":tp,"fp":fp,"truth":truth_n,"sensitivity":tp/truth_n if truth_n else 0.0,"fa_per_24h":fp*24/hours if hours else math.inf}


def events_from_choices(actions, choices, forced=False):
    choice_map = {(str(x.group_id), str(x.selected_model)) for x in choices.itertuples(index=False)}
    selected_groups = {x[0] for x in choice_map}
    rows=[]
    for gid, group in actions.groupby("group_id"):
        gid=str(gid); chosen=None
        for _, row in group.iterrows():
            if (gid,str(row.selected_model)) in choice_map: chosen=row; break
        if chosen is None and forced: chosen=group.sort_values("peak_probability",ascending=False).iloc[0]
        if chosen is not None:
            rows.append({"patient":str(chosen.patient),"recording":str(chosen.recording),"event_start":float(chosen.event_start),"event_end":float(chosen.event_end),"peak_probability":float(chosen.peak_probability),"selected_model":str(chosen.selected_model),"group_id":gid})
    return pd.DataFrame(rows)


def main():
    h4=load("h4_card",ROOT/"scripts/84_run_groupkfold_h4_router.py")
    op=Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/scripts/240_run_h4_protocol_matched_oracle.py")
    ora=load("ora_card",op)
    features=pd.read_parquet(SOURCE/"h4_groupkfold_event_features.parquet")
    validation=pd.read_csv(SOURCE/"h4_groupkfold_router_validation.csv")
    seizures=pd.read_parquet(ROOT/"manifests/seizures.parquet")
    recordings=pd.read_parquet(ROOT/"manifests/recordings.parquet")
    durations=recordings.set_index("recording_id").duration_s.astype(float).to_dict()
    assignments=pd.read_csv(ORACLE/"oracle_assignments.csv")
    linkage=pd.read_csv(AUDIT/"h1_h4_group_linkage.csv")
    rows=[]
    for (fold,seed),frame in features[features.partition.astype(str)=="test"].groupby(["split","seed"]):
        fold,seed=int(fold),int(seed); starts,patients=h4._evaluation_meta(frame); actions=ora.action_table(frame,h4,MODELS)
        # Best single uses the original validation-locked expert.
        vr=validation[(validation.fold==fold)&(validation.seed==seed)&(validation.method=="best_single")].iloc[0]
        best=h4._event_rows_from_model(frame,str(vr.selected_model))
        rows.append({"fold":fold,"seed":seed,"policy":"best_single",**score(best,starts,patients,seizures,durations,h4)})
        assign=assignments[(assignments.fold==fold)&(assignments.seed==seed)]
        full=events_from_choices(actions,assign,False); forced=events_from_choices(actions,assign,True)
        rows.append({"fold":fold,"seed":seed,"policy":"full_abstaining_oracle",**score(full,starts,patients,seizures,durations,h4)})
        rows.append({"fold":fold,"seed":seed,"policy":"forced_selection_oracle",**score(forced,starts,patients,seizures,durations,h4)})
        for policy in ("router_model_aware","router_no_model_id"):
            pick=linkage[(linkage.fold==fold)&(linkage.seed==seed)&(linkage.router==policy)&(linkage.retained)][["group_id","selected_model"]].drop_duplicates()
            event=events_from_choices(actions,pick,False)
            rows.append({"fold":fold,"seed":seed,"policy":policy,**score(event,starts,patients,seizures,durations,h4)})
    out=pd.DataFrame(rows); out.to_csv(AUDIT/"cardinality_first_matcher_per_run.csv",index=False)
    summary=out.groupby("policy",as_index=False)[["sensitivity","fa_per_24h"]].mean(); summary.to_csv(AUDIT/"cardinality_first_matcher_summary.csv",index=False)
    manifest={"analysis":"cardinality-first matcher robustness","status":"post-hoc_no_retuning","runs":15,"summary":summary.to_dict(orient="records")}
    (AUDIT/"cardinality_first_matcher_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(summary.to_string(index=False))

if __name__=="__main__": main()
