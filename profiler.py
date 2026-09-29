"""
profiler.py - Understand ANY table without knowing its columns in advance.

* parse_numeric()   turns messy strings into numbers:
                    "₹ 1,250.50", "Rs 31/kg", "2.5 %", "7 days", "1.2 lakh", "3k", "(45)", "10-12"
* map_ordinal()     turns Yes/No, Low/Medium/High, Poor..Excellent into numbers
* profile_table()   infers each column's kind and statistics - this profile (not the raw
                    data) is what the LLM "explorer" sees.
Nothing here depends on a particular dataset or column name.
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from safety import sanitise

NA_STRINGS = {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "nil", "not available", "#n/a", "tbd"}

_MULTIPLIERS = [  # (regex on the lower-cased text, factor) - a number must come right before the word
    (r"\d\s*(?:crores?|cr)\b", 1e7), (r"\d\s*(?:lakhs?|lacs?)\b", 1e5), (r"\d\s*(?:billion|bn)\b", 1e9),
    (r"\d\s*(?:million|mn)\b", 1e6), (r"\d\s*(?:thousand|k)\b", 1e3),
]
_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
# A value "looks numeric" if it is one number (or a range), optionally with a currency prefix and a
# short unit suffix: "Rs 31/kg", "₹ 1,250", "2.5 %", "7 days", "1.2 lakh", "10-12". "ISO 9001" does not.
_LOOKS_NUMERIC = (r"\s*(?:rs\.?|inr|usd|eur|gbp|\$|₹|€|£)?\s*[-+(]?\s*\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?\)?"
                  r"\s*(?:(?:-|–|to)\s*\d[\d,]*(?:\.\d+)?)?\s*(?:%|[a-z/.₹$ ]{0,16})\s*")

ORDINAL_VOCABS = [
    {"no": 0, "yes": 1, "n": 0, "y": 1, "false": 0, "true": 1, "0": 0, "1": 1},
    {"low": 1, "medium": 2, "med": 2, "moderate": 2, "high": 3, "very high": 4, "very low": 0},
    {"very poor": 1, "poor": 2, "fair": 3, "average": 3, "avg": 3, "good": 4, "very good": 5, "excellent": 6},
    {"bad": 1, "ok": 2, "okay": 2, "good": 3, "great": 4},
]

ID_NAME_HINTS = ("id", "code", "no", "num", "number", "key", "sku", "ref", "gstin", "pan")
NAME_HINTS = ("name", "vendor", "supplier", "company", "party", "firm", "seller", "provider", "brand", "model")
GROUP_HINTS = ("material", "category", "product", "item", "commodity", "segment", "part", "service", "type",
               "class", "group", "family", "grade")


CODE_TOKENS = {"id", "code", "no", "num", "number", "nbr", "sku", "zip", "pin", "pincode", "phone", "mobile",
               "po", "invoice", "ref", "key", "gstin", "pan", "hsn"}
ENTITY_TOKENS = {"store", "brand", "vendor", "supplier", "item", "product", "class", "classification", "category",
                 "branch", "plant", "warehouse", "location", "dept", "department", "region", "customer", "party",
                 "inventory", "sap", "material", "site", "outlet", "shop", "account", "batch", "lot"}


def looks_like_code(col: str) -> bool:
    """Integer columns that NAME things rather than MEASURE them: store no., brand code, PO number."""
    toks = [t for t in str(col).lower().split("_") if t]
    if not toks:
        return False
    return toks[-1] in CODE_TOKENS or set(toks) <= (ENTITY_TOKENS | CODE_TOKENS)


def has_token(col: str, tokens) -> bool:
    """Whole-word match on snake_case names: 'vendor_id' has 'id'; 'notes' does not have 'no'."""
    parts = set(str(col).lower().split("_"))
    return any(t in parts for t in tokens)


# ---------------------------------------------------------------- header helpers
def normalise_name(name: str) -> str:
    s = str(name).strip()
    # split camelCase / PascalCase: "VendorNumber" -> "vendor_number", "PONumber" -> "po_number", "onHand" -> "on_hand"
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", s).lower()
    s = s.replace("%", " pct ").replace("₹", " inr ").replace("$", " usd ").replace("#", " no ")
    s = re.sub(r"[^0-9a-z]+", "_", s).strip("_")
    if not s:
        s = "col"
    if s[0].isdigit():
        s = "c_" + s
    return s


def normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """snake_case, unique column names. Returns (df, {new_name: original_header})."""
    seen, new_cols, originals = {}, [], {}
    for c in df.columns:
        base = normalise_name(c)
        name, i = base, 2
        while name in seen:
            name, i = f"{base}_{i}", i + 1
        seen[name] = True
        new_cols.append(name)
        originals[name] = str(c).strip()
    out = df.copy()
    out.columns = new_cols
    return out, originals


# ---------------------------------------------------------------- value parsing
def _clean_text(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip()
    return t.mask(t.str.lower().isin(NA_STRINGS))


def parse_numeric(s: pd.Series) -> pd.Series:
    """Vectorised messy-number parser. Unparseable -> NaN. Never raises."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    t = _clean_text(s).str.lower()
    nn = t.dropna()
    # European decimal comma ("12,5"): 1-2 digits after a comma and no dots anywhere -> comma is the decimal mark
    if len(nn) and nn.str.fullmatch(r"\s*-?\d+,\d{1,2}\s*").mean() >= 0.3 and not nn.str.contains(".", regex=False).any():
        t = t.str.replace(",", ".", regex=False)
    t = t.where(t.str.fullmatch(_LOOKS_NUMERIC, na=False))
    # accounting negatives: (45) -> -45
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    factor = pd.Series(1.0, index=s.index)
    for pat, f in _MULTIPLIERS:
        hit = t.str.contains(pat, regex=True, na=False)
        factor = factor.where(~hit, f)
    t = t.str.replace(",", "", regex=False)
    # ranges "10-12" / "10 to 12" -> midpoint
    rng = t.str.extract(rf"^\s*[^\d\-+]*({_NUM_RE})\s*(?:-|–|to)\s*({_NUM_RE})\b")
    def num(x):
        return pd.to_numeric(x, errors="coerce").astype("float64")
    first = num(t.str.extract(f"({_NUM_RE})")[0])
    lo, hi = num(rng[0]), num(rng[1])
    both = lo.notna() & hi.notna()
    mid = (lo + hi) / 2
    val = first.where(~both, mid)
    return (val * factor).astype(float)


def map_ordinal(s: pd.Series, min_share: float = 0.9) -> tuple[pd.Series | None, dict | None]:
    """If (almost) all values belong to a known ordinal vocabulary, map them to numbers."""
    t = _clean_text(s).str.lower().str.replace(r"\s+", " ", regex=True)
    vals = t.dropna()
    if vals.empty or vals.nunique() > 8:
        return None, None
    for vocab in ORDINAL_VOCABS:
        share = vals.isin(vocab.keys()).mean()
        if share >= min_share and vals.nunique() >= 2:
            return t.map(vocab).astype(float), {k: v for k, v in vocab.items() if k in set(vals)}
    return None, None


def parse_dates(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return None
    t = _clean_text(s).dropna()
    if t.empty or not t.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{4}-\d{2}", regex=True).mean() > 0.8:
        return None
    txt = _clean_text(s)
    iso = txt.dropna().str.match(r"^\s*\d{4}-\d{1,2}-\d{1,2}").mean() >= 0.8
    parsed = pd.to_datetime(txt, errors="coerce", dayfirst=not iso, format="ISO8601" if iso else "mixed")
    return parsed if parsed.notna().mean() >= 0.8 * max(s.notna().mean(), 1e-9) else None


# ---------------------------------------------------------------- column understanding
def to_numeric_column(s: pd.Series, kind: str) -> pd.Series:
    """Convert a column of a known kind to numbers (used for scoring)."""
    if kind in ("numeric",):
        return parse_numeric(s)
    if kind in ("ordinal", "boolean"):
        mapped, _ = map_ordinal(s, min_share=0.6)
        return mapped if mapped is not None else parse_numeric(s)
    if kind == "date":
        d = parse_dates(s)
        if d is None:
            return pd.Series(np.nan, index=s.index)
        return (pd.Timestamp(datetime.now().date()) - d).dt.days.astype(float)   # age in days
    return parse_numeric(s)


def _unit_hint(raw: pd.Series, header: str) -> str:
    h = header.lower()
    sample = " ".join(raw.dropna().astype(str).head(50).tolist()).lower()
    if "%" in h or "pct" in h or "percent" in h or "%" in sample:
        return "%"
    if any(c in h + sample for c in ("₹", "rs", "inr")):
        return "INR"
    if "$" in h + sample or "usd" in h:
        return "USD"
    if re.search(r"\bdays?\b|_days", h + " " + sample):
        return "days"
    return ""


def profile_column(name: str, s: pd.Series, original: str, n_rows: int) -> dict:
    non_null = _clean_text(s) if not pd.api.types.is_numeric_dtype(s) else s
    nn = int(non_null.notna().sum())
    info = {"column": name, "original_header": original, "non_null": nn,
            "missing_pct": round(100 * (1 - nn / max(n_rows, 1)), 1)}
    vals = non_null.dropna()
    uniq = int(vals.nunique()) if nn else 0
    info["unique"] = uniq
    info["unique_ratio"] = round(uniq / nn, 3) if nn else 0.0
    kind = "empty"
    if nn == 0:
        kind = "empty"
    elif uniq == 1:
        kind = "constant"
    else:
        ordmap, vocab = map_ordinal(s)
        num = parse_numeric(s)
        parse_rate = float(num.notna().sum() / nn) if nn else 0.0
        dates = parse_dates(s) if parse_rate < 0.9 else None
        if vocab is not None:
            kind = "boolean" if set(vocab.values()) <= {0, 1} else "ordinal"
            info["mapping"] = vocab
            num = ordmap
        elif dates is not None:
            kind = "date"
            num = to_numeric_column(s, "date")
            info["date_range"] = [str(dates.min().date()), str(dates.max().date())]
        elif parse_rate >= 0.7:
            is_int_like = bool(((num.dropna() % 1) == 0).all())
            if is_int_like and looks_like_code(name):
                kind = "id" if info["unique_ratio"] >= 0.9 else "code"
            else:
                kind = "numeric"
            info["parse_rate"] = round(parse_rate, 3)
        else:
            avg_len = float(vals.astype(str).str.len().mean())
            if has_token(name, ID_NAME_HINTS) and info["unique_ratio"] >= 0.8 and avg_len <= 40:
                kind = "id"
            elif info["unique_ratio"] >= 0.95 and avg_len <= 40:
                kind = "name" if has_token(name, NAME_HINTS) else (
                    "id" if vals.astype(str).str.fullmatch(r"[A-Za-z]{0,6}[-_ ]?\d{1,9}").mean() > 0.9 else "name")
            elif avg_len > 40:
                kind = "text"
            elif uniq <= max(60, int(0.3 * nn)):
                kind = "categorical"
            else:
                kind = "name" if info["unique_ratio"] > 0.6 else "categorical"
        if kind in ("numeric", "ordinal", "boolean", "date"):
            nv = num.dropna()
            if len(nv):
                info.update(min=_r(nv.min()), p25=_r(nv.quantile(.25)), median=_r(nv.median()),
                            p75=_r(nv.quantile(.75)), max=_r(nv.max()), std=_r(nv.std()))
            info["unit_hint"] = _unit_hint(s, original)
        if kind in ("categorical", "boolean", "ordinal", "code"):
            info["top_values"] = {str(k): int(v) for k, v in vals.astype(str).value_counts().head(6).items()}
    info["kind"] = kind
    info["examples"] = [str(x)[:40] for x in vals.astype(str).drop_duplicates().head(3).tolist()]
    return info


def _r(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(x):
        return None
    return float(f"{x:.4g}")


def profile_table(df: pd.DataFrame, originals: dict | None = None, sample_rows: int = 20000) -> list[dict]:
    """Profile every column. Large tables are profiled on a random sample (fast)."""
    originals = originals or {}
    work = df.sample(sample_rows, random_state=7) if len(df) > sample_rows else df
    return [profile_column(c, work[c], originals.get(c, c), len(work)) for c in df.columns]


NUMERIC_KINDS = ("numeric", "ordinal", "boolean", "date")


def candidate_criteria(profile: list[dict]) -> list[str]:
    return [p["column"] for p in profile if p["kind"] in NUMERIC_KINDS and p.get("missing_pct", 100) < 90]


def profile_for_llm(profile: list[dict], n_rows: int, anonymise: bool = True, max_cols: int = 60) -> dict:
    """Compact, privacy-aware profile for the LLM. Names/IDs/free-text examples are masked when anonymising."""
    cols = []
    for p in profile[:max_cols]:
        q = {k: v for k, v in p.items() if k not in ("unique_ratio",)}
        q["examples"] = [sanitise(e, 40) for e in p.get("examples", [])]
        if "top_values" in q:
            q["top_values"] = {sanitise(k, 40): v for k, v in q["top_values"].items()}
        if anonymise and p["kind"] in ("name", "text", "id"):
            q["examples"] = [f"<{p['kind']} value, {len(e)} chars>" for e in p.get("examples", [])]
        cols.append(q)
    return {"n_rows": n_rows, "n_columns": len(profile), "columns": cols,
            "note": "Examples of names/IDs/free text are masked for privacy." if anonymise else ""}
