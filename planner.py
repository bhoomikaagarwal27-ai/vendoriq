"""
planner.py - The ANALYSIS PLAN: which columns matter, in which direction, how much.

Three sources of a plan, in order of preference:
  1. LLM explorer  -> LLM judge (ai_engine.explore_and_judge)
  2. heuristic_plan() - keyword + statistics rules, used when AI is off (works on any table)
  3. the user, who can edit every row of the plan in the app
Whatever the source, validate_plan() is the final authority: it repairs or rejects anything
that does not fit the actual data (missing columns, text used as a number, ID columns as
criteria, constant columns, bad formulas...). Every repair is logged for the user.

prepare() then cleans the data according to the plan: parses numbers, imputes gaps with the
group median (flagged), removes duplicates, flags extreme values, evaluates derived metrics.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from profiler import NAME_HINTS, GROUP_HINTS, ID_NAME_HINTS, NUMERIC_KINDS, has_token, to_numeric_column
from safety import sanitise

# ---------------------------------------------------------------- heuristic knowledge (generic)
# Checked in this order: strong "lower is better" words, then "higher is better", then weak "lower".
STRONG_LOWER = ("defect", "reject", "complaint", "risk", "delay", "late", "penalty", "incident", "downtime",
                "error", "return", "claim", "breakdown", "failure", "ppm", "scrap")
HIGHER = ("pass", "otif", "on_time", "ontime", "quality", "rating", "score", "uptime", "fill_rate", "accuracy",
          "satisfaction", "nps", "purity", "esg", "certif", "iso", "complian", "capacity", "experience",
          "credit", "payment_term", "warranty", "audit", "years_in", "stock", "availability", "discount")
WEAK_LOWER = ("price", "cost", "rate", "fee", "charge", "tariff", "freight", "distance", "lead", "time", "tat",
              "turnaround", "days", "moq", "minimum_order", "age", "inr", "usd", "amount")
ROLE_WORDS = [  # order matters: specific words first ("pass_rate" is quality, not cost)
    ("risk", ("risk", "complaint", "incident", "penalty", "claim")),
    ("sustainability", ("esg", "sustain", "carbon", "green", "certif", "iso", "complian")),
    ("quality", ("quality", "defect", "reject", "pass", "purity", "ppm", "scrap", "return", "rating", "score", "audit")),
    ("financial", ("credit", "payment", "term")),
    ("delivery", ("lead", "delivery", "delay", "late", "tat", "turnaround", "otif", "on_time", "ontime", "days")),
    ("capacity", ("capacity", "moq", "stock", "availability", "volume")),
    ("cost", ("price", "cost", "rate", "fee", "charge", "tariff", "freight", "amount", "inr", "usd", "discount")),
]
ROLE_WEIGHT = {"cost": 25, "quality": 20, "delivery": 18, "risk": 12, "financial": 10,
               "sustainability": 6, "capacity": 6, "other": 5}
MAX_CRITERIA = 12


def guess_direction(col: str) -> tuple[str, bool]:
    """Returns (direction, confident)."""
    c = col.lower()
    if any(w in c for w in STRONG_LOWER):
        return "lower", True
    if any(w in c for w in HIGHER):
        return "higher", True
    if any(w in c for w in WEAK_LOWER):
        return "lower", True
    return "higher", False


def guess_role(col: str) -> str:
    c = col.lower()
    for role, words in ROLE_WORDS:
        if any(w in c for w in words):
            return role
    return "other"


def _pretty(p: dict) -> str:
    return str(p.get("original_header") or p["column"]).strip()[:40]


def heuristic_plan(profile: list[dict]) -> dict:
    by = {p["column"]: p for p in profile}
    ids = [p for p in profile if p["kind"] == "id"]
    names = [p for p in profile if p["kind"] == "name"]
    entity = next((p["column"] for p in ids if has_token(p["column"], ID_NAME_HINTS)), None) \
        or (ids[0]["column"] if ids else None) or (names[0]["column"] if names else "")
    label = next((p["column"] for p in names if has_token(p["column"], NAME_HINTS)), None) \
        or (names[0]["column"] if names else entity)
    groups = [p for p in profile if p["kind"] == "categorical" and 2 <= p["unique"] <= 40
              and has_token(p["column"], GROUP_HINTS)]
    group = groups[0]["column"] if groups else ""
    criteria, ignored = [], []
    for p in profile:
        c = p["column"]
        if c in (entity, label, group):
            continue
        if p["kind"] in NUMERIC_KINDS and p["missing_pct"] <= 50:
            direction, sure = guess_direction(c)
            role = guess_role(c)
            if not sure and p.get("unit_hint") == "days":   # e.g. "Avg Delivery: 7 days" -> less is better
                direction, sure = "lower", True
            if p["kind"] == "date":          # dates become "age in days": more recent is better
                direction, sure, role = "lower", True, "other"
            weight = ROLE_WEIGHT[role]
            if c.endswith("_total") or c.startswith("records"):
                weight = 5          # summed totals / record counts mostly reflect business volume, not performance
            criteria.append({"column": c, "label": _pretty(p) + (" (age, days)" if p["kind"] == "date" else ""), "direction": direction,
                             "weight": weight, "role": role,
                             "reason": "keyword rule" + ("" if sure else " (direction guessed - please check)")})
        else:
            why = {"text": "free text", "name": "name/label column", "id": "identifier", "code": "code number (names something, does not measure it)",
                   "constant": "same value in every row",
                   "empty": "no data", "categorical": "category (usable as filter or group)"}.get(p["kind"], p["kind"])
            if p["kind"] in NUMERIC_KINDS:
                why = f"{p['missing_pct']}% missing"
            ignored.append({"column": c, "reason": why})
    criteria = sorted(criteria, key=lambda x: -x["weight"])[:MAX_CRITERIA]
    return {"source": "heuristic", "dataset_summary": f"Table with {len(profile)} columns; plan built by keyword rules (AI not used).",
            "entity_column": entity or "", "label_column": label or entity or "", "group_column": group,
            "criteria": criteria, "ignored_columns": ignored, "filter_suggestions": [], "derived_metrics": [],
            "data_quality_notes": [f"{p['column']}: {p['missing_pct']}% missing" for p in profile if 10 <= p.get("missing_pct", 0) < 100][:8],
            "analysis_questions": []}


# ---------------------------------------------------------------- safe formulas for derived metrics
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def check_formula(formula: str, allowed: set[str]) -> tuple[bool, str]:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return False, "not a valid formula"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd, ast.Load) + tuple(_OPS)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                return False, f"unknown column '{node.id}'"
            continue
        return False, f"'{type(node).__name__}' is not allowed (only + - * / numbers and column names)"
    return True, ""


def eval_formula(formula: str, cols: dict[str, pd.Series]) -> pd.Series:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div):
                b = b.replace(0, np.nan) if isinstance(b, pd.Series) else (np.nan if b == 0 else b)
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            return cols[n.id]
        raise ValueError("unsupported")
    out = ev(ast.parse(formula, mode="eval"))
    return out.astype(float) if isinstance(out, pd.Series) else pd.Series(float(out), index=next(iter(cols.values())).index)


# ---------------------------------------------------------------- validation (final authority)
def validate_plan(plan: dict, profile: list[dict]) -> tuple[dict, list[str]]:
    by = {p["column"]: p for p in profile}
    log: list[str] = []
    fixed = {k: v for k, v in plan.items()}
    fallback = heuristic_plan(profile)

    for fld in ("entity_column", "label_column"):
        c = fixed.get(fld) or ""
        if c and c not in by:
            log.append(f"{fld} '{c}' does not exist - replaced with '{fallback[fld]}'.")
            fixed[fld] = fallback[fld]
    ent = fixed.get("entity_column") or ""
    if ent and by[ent]["unique_ratio"] < 0.5:
        log.append(f"entity_column '{ent}' is not unique enough ({by[ent]['unique_ratio']:.0%}) - replaced with '{fallback['entity_column']}'.")
        fixed["entity_column"] = fallback["entity_column"]
    if not fixed.get("label_column"):
        fixed["label_column"] = fixed.get("entity_column", "")
    g = fixed.get("group_column") or ""
    if g and (g not in by or not (2 <= by[g]["unique"] <= max(60, by[g]["non_null"] // 2))):
        log.append(f"group_column '{g}' is unusable (missing or too many/few distinct values) - no grouping.")
        g = ""
    fixed["group_column"] = g

    # derived metrics first, so criteria may reference them
    numeric_cols = {p["column"] for p in profile if p["kind"] in NUMERIC_KINDS}
    derived_ok = []
    for d in (fixed.get("derived_metrics") or [])[:3]:
        name = "".join(ch if ch.isalnum() else "_" for ch in str(d.get("name", "")).lower()).strip("_") or "derived"
        if name in by:
            name = f"{name}_derived"
        ok, why = check_formula(str(d.get("formula", "")), numeric_cols)
        if not ok:
            log.append(f"Derived metric '{d.get('name')}' rejected: {why}.")
            continue
        derived_ok.append({**d, "name": name, "direction": d.get("direction") if d.get("direction") in ("higher", "lower") else "lower"})
    fixed["derived_metrics"] = derived_ok
    derived_names = {d["name"] for d in derived_ok}

    crit, seen = [], set()
    for c in fixed.get("criteria") or []:
        col = str(c.get("column", ""))
        if col in seen:
            continue
        if col not in by and col not in derived_names:
            log.append(f"Criterion '{col}' does not exist - removed.")
            continue
        if col in (fixed["entity_column"], fixed["label_column"], g):
            log.append(f"Criterion '{col}' is the ID/label/group column - removed.")
            continue
        if col in by:
            p = by[col]
            if p["kind"] not in NUMERIC_KINDS:
                log.append(f"Criterion '{col}' is {p['kind']}, not a number - removed.")
                continue
            if p["missing_pct"] > 60:
                log.append(f"Criterion '{col}' is {p['missing_pct']}% empty - removed.")
                continue
            if (p.get("std") in (0, None)) and p["kind"] != "boolean":
                log.append(f"Criterion '{col}' has no variation - removed.")
                continue
        direction = c.get("direction") if c.get("direction") in ("higher", "lower") else guess_direction(col)[0]
        if direction != c.get("direction"):
            log.append(f"Criterion '{col}': direction '{c.get('direction')}' invalid - set to '{direction}'.")
        try:
            w = float(c.get("weight", 0))
        except (TypeError, ValueError):
            w = 0.0
        w = min(max(w, 0.0), 100.0)
        crit.append({"column": col, "label": str(c.get("label") or col)[:40], "direction": direction, "weight": w,
                     "role": c.get("role") or guess_role(col), "reason": str(c.get("reason", ""))[:200]})
        seen.add(col)
    if not crit:
        log.append("No valid criteria in the plan - the keyword plan's criteria are used instead.")
        crit = fallback["criteria"]
    if sum(c["weight"] for c in crit) <= 0:
        log.append("All weights were 0 - equal weights applied.")
        for c in crit:
            c["weight"] = 10.0
    fixed["criteria"] = crit[:MAX_CRITERIA]

    fs = []
    for f in fixed.get("filter_suggestions") or []:
        col, op = str(f.get("column", "")), str(f.get("operator", ""))
        if col in by and op in (">=", "<=", "==", "!=", "contains", "not contains"):
            fs.append({"column": col, "operator": op, "value": str(f.get("value", "")), "reason": str(f.get("reason", ""))[:200]})
        else:
            log.append(f"Filter suggestion on '{col}' ({op}) ignored - column or operator not valid.")
    fixed["filter_suggestions"] = fs
    for k in ("ignored_columns", "data_quality_notes", "analysis_questions"):
        fixed[k] = fixed.get(k) or []
    return fixed, log


# ---------------------------------------------------------------- preparation / cleaning
@dataclass
class Prepared:
    data: pd.DataFrame                 # _id, _label, _group, crit__<col> ..., _flags, _excluded_reason
    criteria: list[dict]
    report: list[dict] = field(default_factory=list)


def _robust_extreme(x: pd.Series, is_pct: bool = False) -> pd.Series:
    """Data-driven sanity flags (no fixed ranges):
    * percentage columns: below 0 or above 100
    * a negative number in a column that is otherwise positive
    * typical unit errors (x100, x1000): small groups - more than 20x (or under 1/20th of) the median;
      groups of 30+ - more than 10x the 99th percentile (or under 1/100th of the 1st percentile),
      which leaves naturally skewed data (capacity, turnover) alone."""
    v = x.dropna()
    flag = pd.Series(False, index=x.index)
    if v.empty:
        return flag
    if is_pct:
        flag |= (x < 0) | (x > 100)
    if (v >= 0).mean() >= 0.8:
        flag |= x < 0
    pos = v[v > 0]
    if len(pos) >= 30:
        hi, lo = pos.quantile(0.99), pos.quantile(0.01)
        flag |= (x > 10 * hi) | ((x > 0) & (x < lo / 100))
    elif len(pos) >= 3:
        med = pos.median()
        flag |= (x > 20 * med) | ((x > 0) & (x < med / 20))
    return flag.fillna(False)


def prepare(df: pd.DataFrame, profile: list[dict], plan: dict, exclude_extremes: bool = True) -> Prepared:
    by = {p["column"]: p for p in profile}
    report: list[dict] = []
    ent, lab, grp = plan.get("entity_column") or "", plan.get("label_column") or "", plan.get("group_column") or ""
    out = pd.DataFrame(index=df.index)
    if ent and ent in df:
        out["_id"] = df[ent].astype("string").str.strip()
    else:
        out["_id"] = pd.NA
    width = max(4, len(str(len(df))))
    missing_id = out["_id"].isna() | (out["_id"] == "")
    if missing_id.any():
        out.loc[missing_id, "_id"] = [f"ROW-{i + 1:0{width}d}" for i in np.where(missing_id)[0]]
        report.append({"step": "IDs", "detail": f"{int(missing_id.sum()):,} rows had no ID - generated ROW-numbers."})
    out["_label"] = (df[lab].astype("string").fillna("").map(lambda t: sanitise(t, 60)) if lab and lab in df else out["_id"])
    out["_group"] = df[grp].astype("string").fillna("(blank)").str.strip() if grp and grp in df else "All"

    # numeric criteria
    needed = {c["column"] for c in plan["criteria"]}
    for d in plan.get("derived_metrics", []):
        needed |= {n.id for n in ast.walk(ast.parse(d["formula"], mode="eval")) if isinstance(n, ast.Name)}
    num_cols: dict[str, pd.Series] = {}
    for col in needed:
        p = by.get(col)
        if p is not None and p["kind"] in NUMERIC_KINDS:
            num_cols[col] = to_numeric_column(df[col], p["kind"])
    for d in plan.get("derived_metrics", []):
        try:
            num_cols[d["name"]] = eval_formula(d["formula"], num_cols)
            report.append({"step": "Derived metric", "detail": f"{d['name']} = {d['formula']}"})
        except Exception as e:  # noqa: BLE001
            report.append({"step": "Derived metric", "detail": f"{d['name']} failed ({type(e).__name__}) - skipped."})

    criteria = []
    for c in plan["criteria"]:
        s = num_cols.get(c["column"])
        if s is None:
            continue
        s = s.replace([np.inf, -np.inf], np.nan)
        miss = float(s.isna().mean())
        if miss > 0.6:
            report.append({"step": "Criterion dropped", "detail": f"{c['label']}: {miss:.0%} missing after parsing."})
            continue
        out[f"crit__{c['column']}"] = s
        criteria.append(c)
    flags = pd.Series("", index=df.index, dtype="object")
    excluded = pd.Series("", index=df.index, dtype="object")

    # rows with too little data
    if criteria:
        cols = [f"crit__{c['column']}" for c in criteria]
        share_missing = out[cols].isna().mean(axis=1)
        too_little = share_missing > 0.5
        excluded[too_little] = "more than half of the criteria are missing"
        if too_little.any():
            report.append({"step": "Rows excluded", "detail": f"{int(too_little.sum()):,} rows miss more than half of the criteria."})
        # extreme values (per group)
        for c in criteria:
            col = f"crit__{c['column']}"
            p = by.get(c["column"], {})
            is_pct = p.get("unit_hint") == "%" or any(w in c["column"] for w in ("pct", "percent"))
            ext = out.groupby("_group", group_keys=False)[col].apply(lambda s: _robust_extreme(s, is_pct)).reindex(out.index).fillna(False)
            if ext.any():
                flags[ext] += f"extreme {c['label']}; "
                report.append({"step": "Implausible values", "detail": f"{c['label']}: {int(ext.sum()):,} value(s) look implausible (unit error, % above 100 or unexpected negative)"
                               + (" - excluded from ranking." if exclude_extremes else " - kept (flagged).")})
                if exclude_extremes:
                    excluded[ext & (excluded == "")] = f"implausible value in {c['label']}"
        # a missing MAJOR criterion (>= 20% of total weight, or the heaviest one) excludes the row;
        # gaps in minor criteria are filled with the group median and flagged
        wsum = sum(max(float(c["weight"]), 0) for c in criteria) or 1
        heaviest = max(criteria, key=lambda c: c["weight"])["column"]
        for c in criteria:
            if c["weight"] / wsum >= 0.2 or c["column"] == heaviest:
                col = f"crit__{c['column']}"
                gap = out[col].isna() & (excluded == "")
                if gap.any():
                    excluded[gap] = f"missing {c['label']} (a major criterion)"
                    report.append({"step": "Rows excluded", "detail": f"{int(gap.sum()):,} row(s) have no value for {c['label']}, a major criterion - not guessed."})
        for c in criteria:
            col = f"crit__{c['column']}"
            gaps = out[col].isna() & (excluded == "")
            if gaps.any():
                fill = out["_group"].map(out[excluded == ""].groupby("_group")[col].median())
                fill = fill.fillna(out.loc[excluded == "", col].median())
                out.loc[gaps, col] = fill[gaps]
                flags[gaps] += f"{c['label']} imputed; "
                report.append({"step": "Imputed", "detail": f"{c['label']}: {int(gaps.sum()):,} blank value(s) filled with the group median."})

    # duplicates within a group
    dup = out.duplicated(subset=["_group", "_id"], keep="first")
    if dup.any():
        excluded[dup & (excluded == "")] = "duplicate ID in the same group"
        report.append({"step": "Duplicates", "detail": f"{int(dup.sum()):,} duplicate ID(s) in the same group - first kept."})
    out["_flags"] = flags.str.strip("; ")
    out["_excluded_reason"] = excluded
    return Prepared(out, criteria, report)


def apply_filters(pool: pd.DataFrame, raw: pd.DataFrame, num_filters: list[dict], text_filters: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """num_filters: [{'column','label','min','max'}] on crit__ columns. text_filters: [{'column','operator','value'}] on raw."""
    reasons = pool["_excluded_reason"].copy()
    for f in num_filters:
        col = f"crit__{f['column']}"
        if col not in pool:
            continue
        if f.get("min") is not None and not pd.isna(f.get("min")):
            bad = (pool[col] < float(f["min"])) & (reasons == "")
            reasons[bad] = pool.loc[bad, col].map(lambda v: f"{f['label']} {v:g} < min {float(f['min']):g}")
        if f.get("max") is not None and not pd.isna(f.get("max")):
            bad = (pool[col] > float(f["max"])) & (reasons == "")
            reasons[bad] = pool.loc[bad, col].map(lambda v: f"{f['label']} {v:g} > max {float(f['max']):g}")
    for f in text_filters:
        col, op, val = f.get("column"), f.get("operator"), str(f.get("value", "")).strip().lower()
        if not col or col not in raw or not op:
            continue
        s = raw.loc[pool.index, col].astype("string").fillna("").str.strip().str.lower()
        num = pd.to_numeric(s, errors="coerce")
        try:
            v_num = float(val)
        except ValueError:
            v_num = None
        if op == "==":
            ok = s == val
        elif op == "!=":
            ok = s != val
        elif op == "contains":
            ok = s.str.contains(val, regex=False)
        elif op == "not contains":
            ok = ~s.str.contains(val, regex=False)
        elif op in (">=", "<=") and v_num is not None:
            ok = (num >= v_num) if op == ">=" else (num <= v_num)
            ok = ok.fillna(False)
        else:
            continue
        bad = (~ok) & (reasons == "")
        reasons[bad] = f"{col} {op} {f.get('value')} not met"
    qualified = pool[reasons == ""].copy()
    excluded = pool[reasons != ""].copy()
    excluded["_excluded_reason"] = reasons[reasons != ""]
    return qualified, excluded
