"""
scoring.py - Deterministic, explainable vendor scoring engine for VendorIQ.

Design principle: the NUMBERS (filters, scores, ranks) are computed here with
transparent rules. The AI model never decides the ranking; it only explains it.
That keeps the ranking stable, auditable and reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Freight assumption: about Rs 4 per tonne-km by road = Rs 0.004 per kg per km
FREIGHT_RS_PER_KG_KM = 0.004
IMPORT_RISK_POINTS = 15        # currency, port and vessel-schedule risk
COMPLAINT_RISK_POINTS = 5      # per quality/service complaint in last 12 months
NO_ISO_RISK_POINTS = 10        # no ISO 9001 quality system

# Minimum purity specification by material (plant QC spec).
PURITY_SPEC = {
    "Methanol": 99.85,
    "Phenol": 99.50,
    "Urea (Technical)": 99.00,
    "Melamine": 99.80,
    "Caustic Soda Flakes": 98.00,
}

CRITERIA = {
    "cost":           {"label": "Landed cost",        "column": "landed_cost",      "better": "lower",  "unit": "Rs/kg"},
    "quality":        {"label": "Quality (QC pass)",  "column": "qc_pass_rate_pct", "better": "higher", "unit": "%"},
    "lead_time":      {"label": "Lead time",          "column": "lead_time_days",   "better": "lower",  "unit": "days"},
    "reliability":    {"label": "On-time delivery",   "column": "otif_pct",         "better": "higher", "unit": "%"},
    "payment":        {"label": "Credit period",      "column": "credit_days",      "better": "higher", "unit": "days"},
    "risk":           {"label": "Supply risk",        "column": "risk_points",      "better": "lower",  "unit": "pts"},
    "sustainability": {"label": "ESG / compliance",   "column": "esg_score",        "better": "higher", "unit": "/100"},
}

PRESETS = {
    "Balanced":                 {"cost": 25, "quality": 20, "lead_time": 15, "reliability": 15, "payment": 10, "risk": 10, "sustainability": 5},
    "Cost-first (L1 mindset)":  {"cost": 50, "quality": 15, "lead_time": 10, "reliability": 10, "payment": 10, "risk": 5,  "sustainability": 0},
    "Quality-critical":         {"cost": 15, "quality": 35, "lead_time": 10, "reliability": 20, "payment": 5,  "risk": 10, "sustainability": 5},
    "Urgent requirement":       {"cost": 15, "quality": 15, "lead_time": 35, "reliability": 25, "payment": 0,  "risk": 10, "sustainability": 0},
    "Working-capital saver":    {"cost": 25, "quality": 15, "lead_time": 10, "reliability": 10, "payment": 30, "risk": 10, "sustainability": 0},
}

TIER_STRONG, TIER_CONSIDER = 65, 45   # scores are relative (min-max), so 65+ is a strong result


def fmt(value, unit: str) -> str:
    """12.5 + '%' -> '12.5%';  30 + 'days' -> '30 days'."""
    v = f"{value:.2f}" if unit == "Rs/kg" else f"{value:g}"
    return f"{v}{unit}" if unit in ("%", "/100") else f"{v} {unit}"


def add_derived_metrics(df: pd.DataFrame, freight_rate: float = FREIGHT_RS_PER_KG_KM) -> pd.DataFrame:
    out = df.copy()
    out["freight_rs_per_kg"] = (out["distance_km"] * freight_rate).round(2)
    out["landed_cost"] = (out["price_inr_per_kg"] + out["freight_rs_per_kg"]).round(2)
    has_iso = out["certifications"].fillna("").str.contains("9001")
    out["cert_count"] = out["certifications"].fillna("").apply(lambda s: len([c for c in s.split(";") if c.strip()]))
    out["risk_points"] = (
        out["complaints_12m"] * COMPLAINT_RISK_POINTS
        + np.where(out["origin"].eq("Import"), IMPORT_RISK_POINTS, 0)
        + np.where(has_iso, 0, NO_ISO_RISK_POINTS)
    ).astype(int)
    return out


def apply_hard_filters(df: pd.DataFrame, qty_mt: float, deadline_days: int, min_purity: float,
                       require_iso: bool, max_landed_cost: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Knock-out rules a procurement expert would apply before any scoring."""
    reasons = []
    for _, r in df.iterrows():
        why = []
        if r["purity_pct"] < min_purity:
            why.append(f"Purity {r['purity_pct']:.2f}% below spec {min_purity:.2f}%")
        if r["lead_time_days"] > deadline_days:
            why.append(f"Lead time {r['lead_time_days']:.0f} d > needed within {deadline_days} d")
        if r["moq_mt"] > qty_mt:
            why.append(f"MOQ {r['moq_mt']:g} MT > required {qty_mt:g} MT")
        if require_iso and "9001" not in str(r["certifications"]):
            why.append("No ISO 9001")
        if max_landed_cost and r["landed_cost"] > max_landed_cost:
            why.append(f"Landed cost Rs {r['landed_cost']:.2f} > budget Rs {max_landed_cost:.2f}")
        reasons.append("; ".join(why))
    df = df.copy()
    df["disqualified_because"] = reasons
    qualified = df[df["disqualified_because"] == ""].drop(columns=["disqualified_because"])
    disqualified = df[df["disqualified_because"] != ""]
    return qualified.reset_index(drop=True), disqualified.reset_index(drop=True)


def normalise(series: pd.Series, better: str) -> pd.Series:
    """Min-max scale to 0..1 where 1 is always 'best'. Equal values -> 1 for all."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    scaled = (series - lo) / (hi - lo)
    return 1 - scaled if better == "lower" else scaled


def weights_to_pct(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be above zero.")
    return {k: v / total for k, v in weights.items()}


def score_vendors(qualified: pd.DataFrame, weights: dict) -> pd.DataFrame:
    if qualified.empty:
        return qualified.copy()
    w = weights_to_pct(weights)
    out = qualified.copy()
    total = pd.Series(0.0, index=out.index)
    for key, meta in CRITERIA.items():
        out[f"norm_{key}"] = normalise(out[meta["column"]], meta["better"])
        out[f"pts_{key}"] = (out[f"norm_{key}"] * w.get(key, 0) * 100).round(2)
        total += out[f"pts_{key}"]
    out["score"] = total.round(1)
    out = out.sort_values(["score", "landed_cost"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out["tier"] = np.select(
        [out["score"] >= TIER_STRONG, out["score"] >= TIER_CONSIDER],
        ["Strong fit", "Consider"], default="Weak fit")
    return out


def stability_analysis(qualified: pd.DataFrame, weights: dict, runs: int = 500,
                       spread: float = 0.25, seed: int = 42) -> pd.DataFrame:
    """Monte-Carlo check: jiggle every weight by +/- spread and count how often
    each vendor still comes out #1. Fixed seed -> same answer every time."""
    if qualified.empty:
        return pd.DataFrame(columns=["vendor_id", "win_share_pct"])
    rng = np.random.default_rng(seed)
    keys = list(CRITERIA.keys())
    base = np.array([weights.get(k, 0) for k in keys], dtype=float)
    norm = np.column_stack([normalise(qualified[CRITERIA[k]["column"]], CRITERIA[k]["better"]).values for k in keys])
    wins = np.zeros(len(qualified))
    for _ in range(runs):
        w = base * rng.uniform(1 - spread, 1 + spread, size=len(keys))
        if w.sum() == 0:
            continue
        s = norm @ (w / w.sum())
        wins[int(np.argmax(s))] += 1
    res = pd.DataFrame({"vendor_id": qualified["vendor_id"].values,
                        "win_share_pct": (wins / runs * 100).round(1)})
    return res.sort_values("win_share_pct", ascending=False).reset_index(drop=True)


def l1_cross_check(scored: pd.DataFrame, qty_mt: float) -> dict:
    """Compare the weighted #1 with the classic L1 rule (lowest landed cost)."""
    if scored.empty:
        return {}
    top = scored.iloc[0]
    l1 = scored.loc[scored["landed_cost"].idxmin()]
    premium_per_kg = round(top["landed_cost"] - l1["landed_cost"], 2)
    return {
        "top_id": top["vendor_id"], "l1_id": l1["vendor_id"],
        "same": top["vendor_id"] == l1["vendor_id"],
        "premium_per_kg": premium_per_kg,
        "premium_total_rs": round(premium_per_kg * qty_mt * 1000, 0),
        "premium_pct": round(premium_per_kg / l1["landed_cost"] * 100, 1) if l1["landed_cost"] else 0,
        "what_you_get": _advantages(top, l1),
    }


def _advantages(a: pd.Series, b: pd.Series) -> list[str]:
    """Plain-language list of where vendor a beats vendor b."""
    out = []
    for key, meta in CRITERIA.items():
        if key == "cost":
            continue
        col, va, vb = meta["column"], a[meta["column"]], b[meta["column"]]
        better = va < vb if meta["better"] == "lower" else va > vb
        if better:
            out.append(f"{meta['label']}: {fmt(va, meta['unit'])} vs {fmt(vb, meta['unit'])}")
    return out


def decision_label(scored: pd.DataFrame, stability: pd.DataFrame) -> tuple[str, str]:
    """Recommend / Recommend with conditions / Refer to committee."""
    if scored.empty:
        return "Refer to committee", "No vendor meets the hard requirements."
    if len(scored) == 1:
        return "Refer to committee", "Only one vendor qualifies - there is no competition to compare against."
    top = scored.iloc[0]
    win = stability.loc[stability["vendor_id"] == top["vendor_id"], "win_share_pct"]
    win = float(win.iloc[0]) if len(win) else 0.0
    risky = top["risk_points"] >= 25 or top["otif_pct"] < 85
    if win >= 70 and top["tier"] == "Strong fit" and not risky:
        return "Recommend", f"{top['vendor_id']} stays #1 in {win:.0f}% of weight variations."
    if win >= 50:
        return "Recommend with conditions", f"{top['vendor_id']} is #1 in {win:.0f}% of weight variations" + (
            "; it also carries a notable supply-risk flag." if risky else "; review the runner-up too.")
    return "Refer to committee", f"Close call: {top['vendor_id']} is #1 in only {win:.0f}% of weight variations."


def rule_based_summary(scored: pd.DataFrame, l1: dict, stability: pd.DataFrame, decision: tuple[str, str]) -> dict:
    """Offline fallback that produces the same structure as the AI output,
    using only templates and numbers. Used when the AI is unavailable."""
    if scored.empty:
        return {"recommended_vendor_id": "", "headline": "No vendor meets the requirements.",
                "why": [], "risks": [], "negotiation_levers": [], "l1_comparison": "",
                "confidence": "Low", "data_gaps": ["Relax a hard filter (deadline, MOQ, purity or budget)."],
                "disagreement_note": ""}
    top = scored.iloc[0]
    why = []
    if len(scored) > 1:
        second = scored.iloc[1]
        adv = _advantages(top, second)
        if top["landed_cost"] < second["landed_cost"]:
            adv.insert(0, f"Landed cost: Rs {top['landed_cost']:g} vs Rs {second['landed_cost']:g} per kg")
        why = [f"Beats runner-up {second['vendor_id']} on {a}" for a in adv[:3]]
    why.append(f"Weighted score {top['score']:.1f}/100 ({top['tier']}).")
    risks = []
    for _, r in scored.head(3).iterrows():
        weakest = min(CRITERIA, key=lambda k: r[f"norm_{k}"])
        m = CRITERIA[weakest]
        risks.append({"vendor_id": r["vendor_id"],
                      "risk": f"Weakest on {m['label']} ({fmt(r[m['column']], m['unit'])})."})
    levers = []
    if not l1.get("same", True):
        levers.append(f"Ask {top['vendor_id']} to narrow the Rs {l1['premium_per_kg']:g}/kg gap to L1 vendor {l1['l1_id']}.")
    best_credit = scored["credit_days"].max()
    if top["credit_days"] < best_credit:
        levers.append(f"Request credit of {best_credit:g} days (best offer in list) instead of {top['credit_days']:g}.")
    if top["moq_mt"] > scored["moq_mt"].min():
        levers.append(f"Negotiate MOQ below {top['moq_mt']:g} MT.")
    if not levers:
        levers.append("Propose a 6-month rate contract to lock the current price.")
    l1_text = ("The top vendor is also the lowest landed-cost (L1) vendor." if l1.get("same")
               else f"L1 vendor is {l1.get('l1_id')}; choosing {top['vendor_id']} costs Rs {l1.get('premium_per_kg', 0):g}/kg "
                    f"({l1.get('premium_pct', 0):g}%) more.")
    conf = {"Recommend": "High", "Recommend with conditions": "Medium"}.get(decision[0], "Low")
    return {"recommended_vendor_id": top["vendor_id"],
            "headline": f"{decision[0]}: {top['vendor_id']} ({top['vendor_name']}). {decision[1]}",
            "why": why, "risks": risks, "negotiation_levers": levers, "l1_comparison": l1_text,
            "confidence": conf, "data_gaps": [], "disagreement_note": ""}
