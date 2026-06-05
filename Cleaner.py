"""
Cleaner.py
AI Data Cleaning Engine — removes duplicates, fills nulls, normalises
casing, standardises date formats, and flags outliers / negative values.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


# ──────────────────────────────────────────────
# DATE HELPERS
# ──────────────────────────────────────────────

_DATE_FORMATS = [
    "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
    "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d-%b-%y", "%d-%b-%Y",            # 15-Mar-24 / 15-Mar-2024
    "%b %d, %Y", "%B %d, %Y",
]


def _parse_date(val: str) -> pd.Timestamp | None:
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(val, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(val, infer_datetime_format=True, dayfirst=True)
    except Exception:
        return None


def _looks_like_date_column(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(20)
    date_pattern = re.compile(
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        r"|(\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4})"
        r"|(\d{4}[-/]\d{2}[-/]\d{2})"
    )
    return sample.str.match(date_pattern).mean() > 0.5


def _standardise_dates(series: pd.Series) -> Tuple[pd.Series, int]:
    """Convert all parseable date strings to ISO YYYY-MM-DD."""
    converted = series.astype(str).apply(
        lambda v: (_parse_date(v).strftime("%Y-%m-%d")
                   if _parse_date(v) is not None else v)
        if pd.notna(v) and v not in ("nan", "NaT") else v
    )
    changed = int((converted != series.astype(str)).sum())
    return converted, changed


# ──────────────────────────────────────────────
# MAIN CLEAN FUNCTION
# ──────────────────────────────────────────────

def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    report: Dict[str, Any] = {
        "duplicates_removed": 0,
        "nulls_filled": 0,
        "casing_normalised": 0,
        "dates_standardised": 0,
        "cleaned_rows": 0,
        "negative_values": [],
        "outliers_flagged": [],
    }

    # ── 1. Remove duplicates ──────────────────────────────
    initial_rows = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = initial_rows - len(df)

    # ── 2. Normalise text casing (title-case names etc.) ──
    for col in df.select_dtypes(include="object").columns:
        if _looks_like_date_column(df[col]):
            continue                     # skip date-like columns here
        original = df[col].copy()
        # Lowercase first, then title-case so "JOHN" → "John"
        df[col] = df[col].where(df[col].isnull(), df[col].astype(str).str.strip().str.title())
        changed = int((df[col].fillna("") != original.fillna("")).sum())
        report["casing_normalised"] += changed

    # ── 3. Standardise date formats ───────────────────────
    for col in df.select_dtypes(include="object").columns:
        if _looks_like_date_column(df[col]):
            df[col], changed = _standardise_dates(df[col])
            report["dates_standardised"] += changed

    # ── 4. Fill missing values ────────────────────────────
    missing_before = int(df.isnull().sum().sum())

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")

    report["nulls_filled"] = missing_before - int(df.isnull().sum().sum())

    # ── 5. Flag negatives ─────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        neg = int((df[col] < 0).sum())
        if neg:
            report["negative_values"].append({"column": col, "count": neg})

    # ── 6. Flag outliers (IQR) ────────────────────────────
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        if len(outliers):
            report["outliers_flagged"].append(
                {"column": col, "count": int(len(outliers))}
            )

    report["cleaned_rows"] = len(df)
    return df, report
