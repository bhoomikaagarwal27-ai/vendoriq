"""
scoring.py - Generic, explainable multi-criteria ranking (works for any criteria list).

score = sum over criteria of  weight% x normalised value x 100
normalised value: min-max within the comparison pool, 1 = best (direction-aware)
Python computes every number; the AI only explains them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIER_STRONG, TIER_CONSIDER = 65, 45      # scores are relative (min-max), so 65+ is a strong result
STABILITY_CANDIDATES = 50                # only the top-N can realistically win; keeps it fast on big data
PARETO_CANDIDATES = 2000


def ccol(c: dict) -> str:
    return f"crit__{c['column']}"


def weights_pct(criteria: list[dict]) -> dict:
    total = sum(max(float(c["weight"]), 0) for c in criteria)
    if total <= 0:
        raise ValueError("At least one weight must be above zero.")
    return {c["column"]: max(float(c["weight"]), 0) / total for c in criteria}


def normalise(s: pd.Series, direction: str) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(1.0, index=s.index)
    x = (s - lo) / (hi - lo)
    return 1 - x if direction == "lower" else x


def score(pool: pd.DataFrame, criteria: list[dict]) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    w = weights_pct(criteria)
    out = pool.copy()
    total = pd.Series(0.0, index=out.index)
    for c in criteria:
        k = c["column"]
        out[f"norm__{k}"] = normalise(out[ccol(c)], c["direction"])
        out[f"pts__{k}"] = (out[f"norm__{k}"] * w[k] * 100).round(2)
        total += out[f"pts__{k}"]
    out["score"] = total.round(1)
    out = out.sort_values(["score", "_id"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["tier"] = np.select([out["score"] >= TIER_STRONG, out["score"] >= TIER_CONSIDER],
                            ["Strong fit", "Consider"], default="Weak fit")
    return out


def stability(scored: pd.DataFrame, criteria: list[dict], runs: int = 500, spread: float = 0.25,
              seed: int = 42) -> pd.DataFrame:
    """Jiggle every weight by +/-25% 500 times; how often does each option stay #1?"""
    if scored.empty:
        return pd.DataFrame(columns=["_id", "win_share_pct"])
    top = scored.head(STABILITY_CANDIDATES)
    keys = [c["column"] for c in criteria]
    base = np.array([max(float(c["weight"]), 0) for c in criteria])
    norm = np.column_stack([normalise(scored[ccol(c)], c["direction"]).loc[top.index].values for c in criteria])
    rng = np.random.default_rng(seed)
    W = base[:, None] * rng.uniform(1 - spread, 1 + spread, size=(len(keys), runs))
    W = W / np.where(W.sum(axis=0) == 0, 1, W.sum(axis=0))
    winners = np.argmax(norm @ W, axis=0)
    wins = np.bincount(winners, minlength=len(top))
    res = pd.DataFrame({"_id": top["_id"].values, "win_share_pct": (wins / runs * 100).round(1)})
    return res.sort_values("win_share_pct", ascending=False).reset_index(drop=True)


def pareto_front(scored: pd.DataFrame, criteria: list[dict]) -> pd.Series:
    """True for options that no other option beats on every criterion at once."""
    flag = pd.Series(False, index=scored.index)
    if scored.empty:
        return flag
    top = scored.head(PARETO_CANDIDATES)
    M = np.column_stack([(top[ccol(c)] * (1 if c["direction"] == "higher" else -1)).values for c in criteria])
    n = len(M)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        ge = (M >= M[i]).all(axis=1)
        gt = (M > M[i]).any(axis=1)
        if (ge & gt).any():
            dominated[i] = True
    flag.loc[top.index] = ~dominated
    return flag


def rule_of_thumb(scored: pd.DataFrame, criteria: list[dict]) -> dict:
    """Compare the weighted #1 with the classic single-criterion rule
    (cheapest if a cost criterion exists - the 'L1' rule - otherwise best on the heaviest criterion)."""
    if scored.empty:
        return {}
    cost = [c for c in criteria if c.get("role") == "cost"]
    rule_c = max(cost, key=lambda c: c["weight"]) if cost else max(criteria, key=lambda c: c["weight"])
    col = ccol(rule_c)
    best_idx = scored[col].idxmin() if rule_c["direction"] == "lower" else scored[col].idxmax()
    top, best = scored.iloc[0], scored.loc[best_idx]
    gap = float(top[col] - best[col])
    ref = float(best[col]) if not pd.isna(best[col]) else 0.0
    pct = abs(gap) / abs(ref) * 100 if ref != 0 else None
    return {"rule": ("lowest " if rule_c["direction"] == "lower" else "highest ") + rule_c["label"],
            "rule_is_l1": rule_c.get("role") == "cost", "criterion": rule_c["label"],
            "top_id": top["_id"], "rule_id": best["_id"], "same": top["_id"] == best["_id"],
            "top_value": _r(top[col]), "rule_value": _r(best[col]), "gap": _r(gap),
            "gap_pct": _r(pct) if pct is not None else None,
            "what_you_get": advantages(top, best, criteria, skip=rule_c["column"])}


def advantages(a: pd.Series, b: pd.Series, criteria: list[dict], skip: str | None = None) -> list[str]:
    out = []
    for c in criteria:
        if c["column"] == skip:
            continue
        va, vb = a[ccol(c)], b[ccol(c)]
        if (va < vb) if c["direction"] == "lower" else (va > vb):
            out.append(f"{c['label']}: {_r(va):g} vs {_r(vb):g}")
    return out


def decision_label(scored: pd.DataFrame, stab: pd.DataFrame, criteria: list[dict]) -> tuple[str, str]:
    if scored.empty:
        return "Refer to committee", "No option meets the requirements."
    if len(scored) == 1:
        return "Refer to committee", "Only one option qualifies - there is nothing to compare it with."
    top = scored.iloc[0]
    win = stab.loc[stab["_id"] == top["_id"], "win_share_pct"]
    win = float(win.iloc[0]) if len(win) else 0.0
    w = weights_pct(criteria)
    weak = [c["label"] for c in criteria if w[c["column"]] >= 0.15 and top.get(f"norm__{c['column']}", 1) < 0.25]
    if win >= 70 and top["tier"] == "Strong fit" and not weak:
        return "Recommend", f"{top['_id']} stays #1 in {win:.0f}% of weight variations."
    if win >= 50:
        return "Recommend with conditions", (f"{top['_id']} is #1 in {win:.0f}% of weight variations"
                                             + (f"; it is weak on {', '.join(weak)}." if weak else "; review the runner-up too."))
    return "Refer to committee", f"Close call: {top['_id']} is #1 in only {win:.0f}% of weight variations."


def _r(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(x):
        return None
    return float(f"{x:.4g}")
