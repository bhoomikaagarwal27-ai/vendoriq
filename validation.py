"""
validation.py - Input validation layer for VendorIQ.

Every uploaded file passes through validate_vendor_data() BEFORE any scoring
or any call to the AI model. Bad rows are either rejected (critical problems)
or repaired and flagged (minor problems), and every decision is logged in an
"issues" table that the user can see.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

MAX_ROWS = 2000
MAX_FILE_MB = 2

REQUIRED_COLUMNS = [
    "vendor_id", "vendor_name", "material", "price_inr_per_kg",
    "lead_time_days", "otif_pct", "qc_pass_rate_pct",
]
OPTIONAL_DEFAULTS = {
    "location": "Not given",
    "origin": "Domestic",
    "purity_pct": None,          # imputed with material median
    "credit_days": None,         # imputed with material median
    "moq_mt": 0,
    "distance_km": None,         # imputed with material median
    "complaints_12m": 0,
    "certifications": "",
    "esg_score": None,           # imputed with material median
    "notes": "",
}
ALL_COLUMNS = REQUIRED_COLUMNS + list(OPTIONAL_DEFAULTS.keys())

# field: (min, max, is_critical)
NUMERIC_RULES = {
    "price_inr_per_kg": (0.01, 100000, True),
    "lead_time_days": (0, 365, True),
    "otif_pct": (0, 100, True),
    "qc_pass_rate_pct": (0, 100, True),
    "purity_pct": (0, 100, False),
    "credit_days": (0, 180, False),
    "moq_mt": (0, 10000, False),
    "distance_km": (0, 5000, False),
    "complaints_12m": (0, 1000, False),
    "esg_score": (0, 100, False),
}

# Phrases that look like instructions aimed at the AI (prompt injection).
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) instructions",
    r"disregard (all |any )?(the )?(previous|prior|above) ",
    r"you are now",
    r"system prompt",
    r"act as ",
    r"recommend this vendor",
    r"rank (this|me|us) (as )?(number one|#?1|first)",
    r"developer mode",
    r"jailbreak",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

COLUMN_ALIASES = {
    "vendor id": "vendor_id", "id": "vendor_id", "supplier_id": "vendor_id",
    "vendor": "vendor_name", "supplier": "vendor_name", "supplier_name": "vendor_name",
    "price": "price_inr_per_kg", "rate": "price_inr_per_kg", "price_per_kg": "price_inr_per_kg",
    "lead_time": "lead_time_days", "leadtime": "lead_time_days",
    "otif": "otif_pct", "on_time_delivery": "otif_pct",
    "qc_pass_rate": "qc_pass_rate_pct", "quality": "qc_pass_rate_pct",
    "purity": "purity_pct", "credit": "credit_days", "payment_terms": "credit_days",
    "moq": "moq_mt", "distance": "distance_km", "complaints": "complaints_12m",
    "esg": "esg_score",
}


@dataclass
class ValidationReport:
    clean: pd.DataFrame
    issues: pd.DataFrame
    rows_in: int = 0
    rows_rejected: int = 0
    rows_repaired: int = 0
    fatal: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.fatal is None and not self.clean.empty


def looks_like_injection(text: str) -> bool:
    return bool(text) and bool(_INJECTION_RE.search(str(text)))


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        key = str(c).strip().lower().replace("-", "_")
        key = COLUMN_ALIASES.get(key, COLUMN_ALIASES.get(key.replace("_", " "), key))
        new_cols[c] = key.replace(" ", "_")
    return df.rename(columns=new_cols)


def validate_vendor_data(raw: pd.DataFrame, file_size_bytes: int | None = None) -> ValidationReport:
    """Validate and clean a vendor table. Never raises; problems go into the report."""
    issues: list[dict] = []

    def log(sev, row, vid, fld, msg):
        issues.append({"severity": sev, "row": row, "vendor_id": vid, "field": fld, "issue": msg})

    empty_issues = pd.DataFrame(columns=["severity", "row", "vendor_id", "field", "issue"])

    if file_size_bytes is not None and file_size_bytes > MAX_FILE_MB * 1024 * 1024:
        return ValidationReport(pd.DataFrame(), empty_issues, fatal=f"File is larger than {MAX_FILE_MB} MB.")
    if raw is None or raw.empty:
        return ValidationReport(pd.DataFrame(), empty_issues, fatal="The file has no rows.")
    if len(raw) > MAX_ROWS:
        return ValidationReport(pd.DataFrame(), empty_issues, rows_in=len(raw),
                                fatal=f"File has {len(raw)} rows; the limit is {MAX_ROWS}.")

    df = _normalise_columns(raw.copy())
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return ValidationReport(pd.DataFrame(), empty_issues, rows_in=len(raw),
                                fatal="Missing required column(s): " + ", ".join(missing)
                                + ". Download the template to see the expected format.")

    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            log("Info", "-", "-", col, "Column not in file; default/imputed values used.")

    df = df[ALL_COLUMNS].copy()
    df["row"] = range(2, len(df) + 2)          # spreadsheet row number (header = row 1)
    df["auto_fixes"] = ""
    reject = pd.Series(False, index=df.index)
    repaired = pd.Series(False, index=df.index)

    # Text fields
    for col in ["vendor_id", "vendor_name", "material", "location", "origin", "certifications", "notes"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["origin"] = df["origin"].replace("", "Domestic").str.title()
    df.loc[~df["origin"].isin(["Domestic", "Import"]), "origin"] = "Domestic"

    for idx, r in df.iterrows():
        if not r["vendor_id"]:
            log("Error", r["row"], "-", "vendor_id", "Blank vendor_id - row rejected.")
            reject[idx] = True
        if not r["material"]:
            log("Error", r["row"], r["vendor_id"], "material", "Blank material - row rejected.")
            reject[idx] = True
        if not r["vendor_name"]:
            df.at[idx, "vendor_name"] = r["vendor_id"] or "Unnamed"
            log("Warning", r["row"], r["vendor_id"], "vendor_name", "Blank name - vendor_id used instead.")

    # Duplicates
    dup = df["vendor_id"].duplicated(keep="first") & (df["vendor_id"] != "")
    for idx in df[dup].index:
        log("Error", df.at[idx, "row"], df.at[idx, "vendor_id"], "vendor_id",
            "Duplicate vendor_id - only the first occurrence is kept.")
        reject[idx] = True

    # Numeric fields: type + range
    for col, (lo, hi, critical) in NUMERIC_RULES.items():
        original = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for idx in df.index:
            val, orig = df.at[idx, col], original[idx]
            vid, row = df.at[idx, "vendor_id"], df.at[idx, "row"]
            if pd.isna(val):
                was_text = not (orig is None or (isinstance(orig, float) and pd.isna(orig)) or str(orig).strip() == "")
                why = f"'{orig}' is not a number" if was_text else "value is blank"
                if critical:
                    log("Error", row, vid, col, f"{why} - row rejected (critical field).")
                    reject[idx] = True
                continue
            if val < lo or val > hi:
                if critical:
                    log("Error", row, vid, col, f"{val:g} is outside the allowed range {lo:g}-{hi:g} - row rejected.")
                    reject[idx] = True
                else:
                    log("Warning", row, vid, col, f"{val:g} is outside {lo:g}-{hi:g}; replaced with material median.")
                    df.at[idx, col] = float("nan")

    # Impute non-critical blanks with the material median
    for col in ["purity_pct", "credit_days", "distance_km", "esg_score", "moq_mt", "complaints_12m"]:
        med_by_mat = df[~reject].groupby("material")[col].median()
        overall = df.loc[~reject, col].median()
        for idx in df[df[col].isna() & ~reject].index:
            fill = med_by_mat.get(df.at[idx, "material"], float("nan"))
            if pd.isna(fill):
                fill = overall if not pd.isna(overall) else 0
            df.at[idx, col] = fill
            repaired[idx] = True
            df.at[idx, "auto_fixes"] += f"{col} imputed; "
            log("Warning", df.at[idx, "row"], df.at[idx, "vendor_id"], col,
                f"Blank value filled with material median ({fill:g}).")

    # Price outliers (common unit error: Rs/MT typed instead of Rs/kg)
    ok_rows = df[~reject]
    med_price = ok_rows.groupby("material")["price_inr_per_kg"].median()
    for idx in ok_rows.index:
        m = med_price.get(df.at[idx, "material"])
        p = df.at[idx, "price_inr_per_kg"]
        if m and p > 5 * m:
            log("Error", df.at[idx, "row"], df.at[idx, "vendor_id"], "price_inr_per_kg",
                f"Price {p:g} is more than 5x the material median ({m:g}). Likely entered per MT, not per kg - row rejected.")
            reject[idx] = True

    # Prompt-injection text in notes
    for idx in df[~reject].index:
        if looks_like_injection(df.at[idx, "notes"]):
            log("Security", df.at[idx, "row"], df.at[idx, "vendor_id"], "notes",
                "Instruction-like text found (possible prompt injection). Notes removed before sending to AI.")
            df.at[idx, "notes"] = "[removed by safety filter]"
            df.at[idx, "auto_fixes"] += "notes sanitised; "
            repaired[idx] = True
        elif len(df.at[idx, "notes"]) > 300:
            df.at[idx, "notes"] = df.at[idx, "notes"][:300] + "..."
            repaired[idx] = True

    clean = df[~reject].drop(columns=["row"]).reset_index(drop=True)
    clean["complaints_12m"] = clean["complaints_12m"].round().astype(int)

    report = ValidationReport(
        clean=clean,
        issues=(pd.DataFrame(issues, columns=["severity", "row", "vendor_id", "field", "issue"])
                .assign(_r=lambda d: pd.to_numeric(d["row"], errors="coerce").fillna(0))
                .sort_values(["_r"], kind="stable").drop(columns="_r").reset_index(drop=True)) if issues else empty_issues,
        rows_in=len(raw),
        rows_rejected=int(reject.sum()),
        rows_repaired=int((repaired & ~reject).sum()),
    )

    counts = clean.groupby("material").size()
    for mat, n in counts.items():
        if n < 2:
            report.warnings.append(f"'{mat}' has only {n} valid vendor - a ranking needs at least 2 to be meaningful.")
    if clean.empty:
        report.fatal = "No valid rows left after validation. See the issues table."
    return report


def validate_requirement(qty_mt: float, deadline_days: int, weights: dict) -> list[str]:
    """Checks on the requirement form. Returns a list of error messages (empty = OK)."""
    errors = []
    if qty_mt is None or qty_mt <= 0:
        errors.append("Required quantity must be greater than 0 MT.")
    elif qty_mt > 10000:
        errors.append("Required quantity above 10,000 MT looks unrealistic for one order - please check units.")
    if deadline_days is None or deadline_days < 1 or deadline_days > 180:
        errors.append("'Needed within' must be between 1 and 180 days.")
    if sum(weights.values()) <= 0:
        errors.append("At least one weight must be above zero.")
    return errors
