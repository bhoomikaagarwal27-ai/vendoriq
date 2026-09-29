"""
aggregate.py - Turn TRANSACTION-level data (many rows per vendor: purchase orders, invoices,
inventory lines) into ONE ROW PER VENDOR, which is what a vendor ranking needs.

  * suggest_key()      finds the column that identifies the vendor (e.g. VendorName, Supplier Code)
  * aggregate_table()  groups by that column:
        - quantities / money totals (qty, units, dollars, amount, freight, onhand ...) -> SUM
        - rates / prices / scores / percentages / days                            -> AVERAGE
        - dates -> latest date;  categories -> most frequent value
        - code columns (store no., brand code, PO number ...) are not measures     -> dropped
        - adds 'records' (number of rows per vendor)
        - adds day gaps between order-type and receipt/payment-type dates, e.g.
          lead_time_days_podate_to_receivingdate, credit_days_invoicedate_to_paydate
Nothing here is specific to one dataset: decisions come from column kinds and name tokens.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from profiler import NUMERIC_KINDS, profile_table, to_numeric_column, parse_dates

VENDOR_TOKENS = ("vendor", "supplier", "party", "seller", "company", "manufacturer", "distributor",
                 "firm", "contractor", "transporter", "provider")
SUM_WORDS = ("total", "qty", "quantity", "units", "dollars", "amount", "sales", "revenue", "spend", "freight",
             "volume", "onhand", "stock", "orders", "value", "count", "cases", "pieces", "weight", "tonnage")
MEAN_WORDS = ("price", "rate", "pct", "percent", "avg", "average", "mean", "rating", "score", "days", "time",
              "per", "ratio", "margin", "yield", "purity", "cost", "grade", "index")
START_DATE_WORDS = ("po", "order", "invoice", "request", "indent", "requisition")
END_DATE_WORDS = ("receiv", "deliver", "arriv", "grn", "ship", "dispatch", "pay")


def _tokens(col: str) -> set:
    return set(re.split(r"_+", col.lower()))


def is_vendor_like(col: str) -> bool:
    c = col.lower()
    return any(t in c for t in VENDOR_TOKENS)


def suggest_key(df: pd.DataFrame, profile: list[dict]) -> str | None:
    """Vendor column with repeats (i.e. rows are transactions). None if data is already one row per vendor."""
    best, best_rank = None, None
    for p in profile:
        c = p["column"]
        if not is_vendor_like(c) or p["kind"] in ("empty", "constant", "text"):
            continue
        n_unique = df[c].nunique(dropna=True)
        if n_unique < 2 or len(df) / max(n_unique, 1) < 1.5:
            continue
        rank = (0 if "name" in c else 1, n_unique)          # prefer the readable name column
        if best_rank is None or rank < best_rank:
            best, best_rank = c, rank
    return best


def _agg_kind(col: str) -> str:
    c = col.lower().replace("_", "")
    if "total" in c:
        return "sum"
    if any(w in c for w in MEAN_WORDS):
        return "mean"
    if any(w in c for w in SUM_WORDS):
        return "sum"
    return "mean"


def _date_gaps(df: pd.DataFrame, profile: list[dict]) -> dict[str, tuple[pd.Series, str, str, str]]:
    dates = {p["column"]: parse_dates(df[p["column"]]) for p in profile if p["kind"] == "date"}
    dates = {k: v for k, v in dates.items() if v is not None}
    out = {}
    for a, da in dates.items():
        if not any(w in a.lower() for w in START_DATE_WORDS):
            continue
        for b, db in dates.items():
            if a == b or not any(w in b.lower() for w in END_DATE_WORDS):
                continue
            credit = "pay" in b.lower()
            if credit and "invoice" not in a.lower():
                continue                        # credit period = invoice -> payment only
            gap = (db - da).dt.days.astype(float)
            med = gap.median()
            if pd.notna(med) and 0 <= med <= 365 and (gap.dropna() >= 0).mean() >= 0.8:
                prefix = "credit_days" if credit else "lead_time_days"
                out[f"{prefix}_{a}_to_{b}"] = (gap, "Credit days" if credit else "Lead time days", a, b)
    return out


def aggregate_table(df: pd.DataFrame, key: str, originals: dict | None = None,
                    profile: list[dict] | None = None) -> tuple[pd.DataFrame, dict, list[str]]:
    """One row per value of `key`. Returns (aggregated df, {column: readable header}, log lines)."""
    originals = originals or {}
    profile = profile or profile_table(df, originals)
    by = {p["column"]: p for p in profile}
    log = []
    k = df[key].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    work = pd.DataFrame({key: k})
    agg, headers = {}, {key: originals.get(key, key)}
    dropped = []
    for c, p in by.items():
        if c == key:
            continue
        kind = p["kind"]
        if kind in NUMERIC_KINDS and kind != "date":
            how = _agg_kind(c)
            work[c] = to_numeric_column(df[c], kind)
            name = f"{c}_{'total' if how == 'sum' else 'avg'}"
            agg[name] = (c, how)
            headers[name] = f"{originals.get(c, c)} ({'total' if how == 'sum' else 'avg'})"
        elif kind == "date":
            pass                                # dates are used through the day gaps below
        elif kind == "categorical":
            work[c] = df[c].astype("string").str.strip()
            headers[c] = originals.get(c, c)
        else:                                   # id, code, name, text, constant, empty
            dropped.append(originals.get(c, c))
    for name, (gap, what, a, b) in _date_gaps(df, profile).items():
        work[name] = gap
        agg[name] = (name, "mean")
        headers[name] = f"{what} ({originals.get(a, a)} → {originals.get(b, b)})"
        log.append(f"Derived '{headers[name]}' from the two dates in each row (average per vendor).")
    work = work[work[key].notna() & (work[key] != "")]
    g = work.groupby(key, sort=False)
    out = g.agg(**agg) if agg else pd.DataFrame(index=g.size().index)
    out["records"] = g.size()
    headers["records"] = "Number of records"
    for c in [c for c in work.columns if c in by and by[c]["kind"] == "categorical" and c != key]:
        vc = work.groupby([key, c], sort=False).size().reset_index(name="n")
        top = vc.sort_values("n", ascending=False).drop_duplicates(key).set_index(key)[c]
        out[c] = top
    out = out.reset_index()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    log.insert(0, f"Rows are transactions: {len(df):,} rows were combined into {len(out):,} vendors (one row per '{originals.get(key, key)}'). "
                  "Quantities and money are summed, prices/rates/scores are averaged.")
    if dropped:
        log.append("Not used as measures (codes / IDs / free text): " + ", ".join(dropped[:12]) + (" …" if len(dropped) > 12 else ""))
    return out, headers, log
