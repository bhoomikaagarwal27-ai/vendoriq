"""
ingest.py - Load one or MANY files of any shape and combine them into one table.

Supported: CSV / TSV / TXT (delimiter auto-detected, any encoding), Excel (.xlsx/.xls, every
sheet), JSON (records or columns), Parquet. Title rows above the real header are skipped.

Combining several tables (auto mode):
  * similar columns (>= 60% overlap)          -> STACK them (rows appended, source_file kept)
  * a shared key column that is unique in each -> JOIN them on that key (like VLOOKUP)
  * otherwise                                  -> stack with the union of columns (and warn)
The user can override the strategy and the join key in the app.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from profiler import NA_STRINGS, normalise_columns

MAX_TOTAL_ROWS = 500_000        # protects memory on the free hosting tier; larger data is sampled
MAX_FILES = 20
MAX_COLUMNS = 300


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    originals: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class CombineResult:
    df: pd.DataFrame
    originals: dict
    strategy: str
    key: str | None
    log: list[str]


# ---------------------------------------------------------------- reading
def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _fix_header(df: pd.DataFrame) -> pd.DataFrame:
    """If the real header is not the first row (title rows in Excel exports), find it."""
    unnamed = sum(str(c).startswith("Unnamed") or str(c).strip() == "" for c in df.columns)
    if unnamed < max(2, 0.5 * len(df.columns)) or df.empty:
        return df
    for i in range(min(15, len(df))):
        row = df.iloc[i]
        filled = row.notna().mean()
        texty = row.dropna().astype(str).str.contains(r"[A-Za-z]").mean() if row.notna().any() else 0
        if filled >= 0.6 and texty >= 0.6:
            new = df.iloc[i + 1:].copy()
            new.columns = [str(x) if pd.notna(x) else f"column_{j + 1}" for j, x in enumerate(row.tolist())]
            return new.reset_index(drop=True)
    return df


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_header(df)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.loc[:, [not (str(c).startswith("Unnamed") and df[c].isna().all()) for c in df.columns]]
    if df.shape[1] > MAX_COLUMNS:
        df = df.iloc[:, :MAX_COLUMNS]
    return df.reset_index(drop=True)


def read_file(name: str, data: bytes) -> list[Table]:
    """Read one uploaded file into one or more tables. Raises ValueError with a readable message."""
    lower = name.lower()
    na = list(NA_STRINGS - {""})
    try:
        if lower.endswith((".xlsx", ".xlsm", ".xls")):
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, na_values=na)
            out = []
            for sheet, df in sheets.items():
                df = _tidy(df)
                if not df.empty and df.shape[1] >= 2:
                    out.append(Table(f"{name} [{sheet}]" if len(sheets) > 1 else name, df))
            if not out:
                raise ValueError("no sheet with a usable table")
            return out
        if lower.endswith(".json"):
            obj = json.loads(_decode(data))
            if isinstance(obj, dict):
                obj = next((v for v in obj.values() if isinstance(v, list)), obj)
            return [Table(name, _tidy(pd.json_normalize(obj) if isinstance(obj, list) else pd.DataFrame(obj)))]
        if lower.endswith(".parquet"):
            return [Table(name, _tidy(pd.read_parquet(io.BytesIO(data))))]
        # CSV / TSV / TXT
        text = _decode(data)
        head = text[:65536]
        try:
            sep = csv.Sniffer().sniff(head, delimiters=[",", ";", "\t", "|"]).delimiter
        except csv.Error:
            sep = "\t" if head.count("\t") > head.count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep, na_values=na, keep_default_na=True,
                         skipinitialspace=True, on_bad_lines="skip", low_memory=False)
        return [Table(name, _tidy(df))]
    except ValueError as e:
        raise ValueError(f"{name}: could not read ({e}).") from e
    except Exception as e:  # noqa: BLE001 - any parser error becomes a user message
        raise ValueError(f"{name}: could not read this file ({type(e).__name__}). "
                         "Save it as CSV (UTF-8) or Excel and try again.") from e


# ---------------------------------------------------------------- combining
def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def find_join_keys(tables: list[pd.DataFrame]) -> list[str]:
    """Columns present in every table and (almost) unique in each - usable as a join key."""
    common = set(tables[0].columns)
    for t in tables[1:]:
        common &= set(t.columns)
    keys = []
    for c in common:
        ok = all(t[c].notna().mean() > 0.8 and t[c].astype(str).nunique() >= 0.9 * t[c].notna().sum() for t in tables)
        if ok:
            keys.append(c)
    keys.sort(key=lambda c: (not any(h in c for h in ("id", "code", "key", "no")), c))
    return keys


def combine(tables: list[Table], strategy: str = "auto", key: str | None = None) -> CombineResult:
    log: list[str] = []
    if not tables:
        raise ValueError("No tables to combine.")
    if len(tables) > MAX_FILES:
        log.append(f"Only the first {MAX_FILES} tables are used.")
        tables = tables[:MAX_FILES]
    normed, originals = [], {}
    for t in tables:
        d, o = normalise_columns(t.df)
        normed.append(d)
        originals.update({k: v for k, v in o.items() if k not in originals})

    if len(normed) == 1:
        df = normed[0]
        strategy = "single"
        log.append(f"1 table: {len(df):,} rows x {df.shape[1]} columns.")
    else:
        colsets = [set(d.columns) for d in normed]
        similar = min(_jaccard(a, b) for i, a in enumerate(colsets) for b in colsets[i + 1:]) >= 0.6
        keys = find_join_keys(normed)
        if strategy == "auto":
            strategy = "stack" if similar else ("join" if keys else "stack")
            log.append(f"Auto-detected strategy: {strategy.upper()} "
                       + ("(files share most columns)." if similar else
                          f"(shared key '{keys[0]}')." if keys else "(no shared key; columns were unioned)."))
        if strategy == "join":
            key = key if key in (keys or []) else (keys[0] if keys else None)
            if key is None:
                log.append("No usable join key found - falling back to STACK.")
                strategy = "stack"
        if strategy == "stack":
            parts = []
            for t, d in zip(tables, normed):
                d = d.copy()
                d["source_file"] = t.name
                parts.append(d)
            df = pd.concat(parts, ignore_index=True, sort=False)
            log.append(f"Stacked {len(parts)} tables -> {len(df):,} rows x {df.shape[1]} columns.")
            originals["source_file"] = "Source file"
        else:
            order = sorted(range(len(normed)), key=lambda i: -len(normed[i]))
            df = normed[order[0]].copy()
            df[key] = df[key].astype(str).str.strip()
            for i in order[1:]:
                right = normed[i].copy()
                right[key] = right[key].astype(str).str.strip()
                if right[key].duplicated().any():
                    num = right.select_dtypes("number").columns.difference([key])
                    agg = {c: ("mean" if c in num else "first") for c in right.columns if c != key}
                    right = right.groupby(key, as_index=False).agg(agg)
                    log.append(f"{tables[i].name}: duplicate keys aggregated before joining.")
                clash = [c for c in right.columns if c in df.columns and c != key]
                stem = normalise_columns(pd.DataFrame(columns=[tables[i].name.split('.')[0]]))[0].columns[0]
                right = right.rename(columns={c: f"{c}_{stem}" for c in clash})
                before = len(df)
                matched = df[key].isin(right[key]).sum()
                df = df.merge(right, on=key, how="left")
                log.append(f"Joined {tables[i].name} on '{key}': {matched:,} of {before:,} rows matched"
                           + (f"; {before - matched:,} had no match (left blank)." if matched < before else "."))

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.append(f"Removed {before - len(df):,} exact duplicate rows.")
    if len(df) > MAX_TOTAL_ROWS:
        log.append(f"Data has {len(df):,} rows; a random sample of {MAX_TOTAL_ROWS:,} is analysed (memory limit).")
        df = df.sample(MAX_TOTAL_ROWS, random_state=42)
    return CombineResult(df.reset_index(drop=True), originals, strategy, key, log)
