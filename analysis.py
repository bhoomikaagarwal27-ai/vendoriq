"""
analysis.py - "Further analysis" computed in Python, chosen by what the data contains.

* trade_offs()      Spearman correlation between criteria, oriented so that + means "good goes
                    with good" (synergy) and - means "good on one = bad on the other" (trade-off)
* group_summary()   median of each criterion per group (only if the plan found a group column)
* criterion_stats() best / median / worst per criterion in the pool
* build_context()   the compact, rounded, privacy-aware package the LLM receives
* rule_based_insights() offline fallback with the same structure as the AI answer
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from safety import sanitise
from scoring import ccol, weights_pct, _r

SCORING_METHOD = ("Each criterion is min-max normalised inside the comparison pool (1 = best), multiplied by its "
                  "weight and summed to 100. Scores are RELATIVE to the options in the pool, not absolute grades.")


def _varying(pool: pd.DataFrame, criteria: list[dict]) -> list[dict]:
    return [c for c in criteria if pool[ccol(c)].nunique(dropna=True) > 1]


def trade_offs(pool: pd.DataFrame, criteria: list[dict], top_n: int = 6) -> list[dict]:
    criteria = _varying(pool, criteria)
    if len(pool) < 5 or len(criteria) < 2:
        return []
    oriented = pd.DataFrame({c["label"]: pool[ccol(c)] * (1 if c["direction"] == "higher" else -1) for c in criteria})
    corr = oriented.corr(method="spearman")
    pairs = []
    labels = list(corr.columns)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.25:
                kind = "trade-off" if r < 0 else "synergy"
                pairs.append({"a": a, "b": b, "rho": _r(r), "type": kind,
                              "meaning": (f"options that are good on {a} tend to be worse on {b}" if r < 0
                                          else f"options good on {a} also tend to be good on {b}")})
    return sorted(pairs, key=lambda p: -abs(p["rho"]))[:top_n]


def correlation_matrix(pool: pd.DataFrame, criteria: list[dict]) -> pd.DataFrame:
    criteria = _varying(pool, criteria)
    oriented = pd.DataFrame({c["label"]: pool[ccol(c)] * (1 if c["direction"] == "higher" else -1) for c in criteria})
    return oriented.corr(method="spearman").round(2)


def criterion_stats(pool: pd.DataFrame, criteria: list[dict]) -> dict:
    out = {}
    for c in criteria:
        s = pool[ccol(c)]
        best = s.min() if c["direction"] == "lower" else s.max()
        worst = s.max() if c["direction"] == "lower" else s.min()
        out[c["label"]] = {"best": _r(best), "median": _r(s.median()), "worst": _r(worst), "better": c["direction"]}
    return out


def group_summary(data: pd.DataFrame, criteria: list[dict], max_groups: int = 12) -> pd.DataFrame:
    live = data[data["_excluded_reason"] == ""]
    if live["_group"].nunique() < 2:
        return pd.DataFrame()
    g = live.groupby("_group")
    tab = pd.DataFrame({"options": g.size()})
    for c in criteria:
        tab[c["label"] + " (median)"] = g[ccol(c)].median().map(_r)
    return tab.sort_values("options", ascending=False).head(max_groups)


def build_context(scored: pd.DataFrame, excluded: pd.DataFrame, criteria: list[dict], plan: dict, group: str,
                  stab: pd.DataFrame, rot: dict, pareto_ids: list, trade: list, report: list,
                  anonymise: bool = True, top_n: int = 10) -> dict:
    """The ONLY data the model sees. All arithmetic is done here, not by the model."""
    w = weights_pct(criteria)
    top = scored.head(top_n)
    first = scored.iloc[0] if not scored.empty else None
    options = []
    for _, r in top.iterrows():
        o = {"id": r["_id"], "rank": int(r["rank"]), "score": float(r["score"]), "tier": r["tier"],
             "values": {c["label"]: _r(r[ccol(c)]) for c in criteria},
             "points": {c["label"]: float(r[f"pts__{c['column']}"]) for c in criteria}}
        if not anonymise:
            o["name"] = sanitise(r["_label"], 60)
        if r.get("_flags"):
            o["data_flags"] = r["_flags"]
        if first is not None and r["_id"] != first["_id"]:
            o["gap_vs_top"] = {c["label"]: _r(r[ccol(c)] - first[ccol(c)]) for c in criteria}
            o["gap_vs_top"]["score"] = _r(r["score"] - first["score"])
        options.append(o)
    return {
        "dataset_summary": sanitise(plan.get("dataset_summary", ""), 300),
        "comparison_pool": {"group": group, "options_ranked": int(len(scored)), "options_excluded": int(len(excluded))},
        "criteria": [{"label": c["label"], "better": c["direction"], "weight_pct": round(w[c["column"]] * 100, 1),
                      "role": c.get("role", "")} for c in criteria],
        "scoring_method": SCORING_METHOD,
        "top_options": options,
        "excluded_examples": [{"id": r["_id"], "reason": r["_excluded_reason"]} for _, r in excluded.head(8).iterrows()],
        "stability_top5": stab.head(5).rename(columns={"_id": "id"}).to_dict(orient="records"),
        "rule_of_thumb_check": rot,
        "pareto_efficient_ids": list(pareto_ids)[:20],
        "trade_offs": trade,
        "criterion_stats": criterion_stats(scored, criteria) if not scored.empty else {},
        "data_cleaning": [f"{x['step']}: {x['detail']}" for x in report][:12],
    }


def rule_based_insights(scored: pd.DataFrame, criteria: list[dict], rot: dict, stab: pd.DataFrame,
                        decision: tuple[str, str], trade: list, pareto_ids: list) -> dict:
    if scored.empty:
        return {"recommended_id": "", "headline": "No option meets the requirements.", "why": [], "risks": [],
                "trade_offs": [], "anomalies": [], "negotiation_levers": [], "further_analysis": [],
                "confidence": "Low", "data_gaps": ["Relax a filter to get candidates."], "disagreement_note": ""}
    top = scored.iloc[0]
    why = []
    if len(scored) > 1:
        second = scored.iloc[1]
        for c in criteria:
            a, b = top[ccol(c)], second[ccol(c)]
            if (a < b) if c["direction"] == "lower" else (a > b):
                why.append(f"Beats runner-up {second['_id']} on {c['label']}: {_r(a):g} vs {_r(b):g}.")
            if len(why) == 3:
                break
    why.append(f"Weighted score {top['score']:.1f}/100 ({top['tier']}).")
    risks = []
    for _, r in scored.head(3).iterrows():
        weakest = min(criteria, key=lambda c: r[f"norm__{c['column']}"])
        risks.append({"id": r["_id"], "risk": f"Weakest on {weakest['label']} ({_r(r[ccol(weakest)]):g})."})
    levers = []
    if rot and not rot.get("same"):
        levers.append(f"Ask {rot['top_id']} to close the gap on {rot['criterion']} ({rot['top_value']:g} vs {rot['rule_value']:g} at {rot['rule_id']}).")
    for c in sorted(criteria, key=lambda c: -c["weight"])[:3]:
        if top[f"norm__{c['column']}"] < 0.6:
            levers.append(f"Negotiate on {c['label']} - {top['_id']} is below the pool's best here.")
    conf = {"Recommend": "High", "Recommend with conditions": "Medium"}.get(decision[0], "Low")
    return {"recommended_id": top["_id"], "headline": f"{decision[0]}: {top['_id']}. {decision[1]}",
            "why": why, "risks": risks,
            "trade_offs": [f"{t['a']} vs {t['b']}: {t['meaning']} (rho {t['rho']})." for t in trade[:3]],
            "anomalies": [], "negotiation_levers": levers or ["Propose a rate contract to lock current terms."],
            "further_analysis": [f"{len(pareto_ids)} option(s) are Pareto-efficient (no other option beats them on every criterion)."],
            "confidence": conf, "data_gaps": [], "disagreement_note": ""}
