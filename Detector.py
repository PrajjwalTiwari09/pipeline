"""
Detector.py
AI Issue Detection Engine — detects missing values, outliers,
duplicates, high cardinality, inconsistent casing, and date format
anomalies.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class Issue:
    severity: str        # High | Medium | Low
    category: str
    field: str
    description: str
    affected_rows: int


@dataclass
class DetectionReport:
    total_rows: int
    total_columns: int
    missing_cells: int
    duplicate_rows: int
    completeness_pct: float
    issues: List[Issue] = field(default_factory=list)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _iqr_bounds(series: pd.Series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _has_mixed_date_formats(series: pd.Series) -> bool:
    """Heuristic: try to find columns that look like dates but have
    multiple format styles (e.g. 01-01-2024 vs 15-Mar-24)."""
    sample = series.dropna().astype(str).head(50)
    import re
    numeric_date = re.compile(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
    alpha_date   = re.compile(r"\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}")
    has_numeric  = sample.str.match(numeric_date).any()
    has_alpha    = sample.str.match(alpha_date).any()
    return bool(has_numeric and has_alpha)


# ──────────────────────────────────────────────
# MAIN DETECT FUNCTION
# ──────────────────────────────────────────────

def detect(df: pd.DataFrame) -> DetectionReport:
    issues: List[Issue] = []

    total_rows    = len(df)
    total_columns = len(df.columns)
    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    total_cells   = total_rows * total_columns

    completeness_pct = round(
        ((total_cells - missing_cells) / total_cells) * 100, 2
    ) if total_cells > 0 else 0.0

    for col in df.columns:

        # ── Missing values ────────────────────────────────
        missing = int(df[col].isnull().sum())
        if missing > 0:
            issues.append(Issue(
                severity="High",
                category="Missing Values",
                field=col,
                description=f"{missing} missing values found",
                affected_rows=missing,
            ))

        # ── Numeric-specific checks ───────────────────────
        if pd.api.types.is_numeric_dtype(df[col]):

            lower, upper = _iqr_bounds(df[col].dropna())
            outlier_count = int(
                ((df[col] < lower) | (df[col] > upper)).sum()
            )
            if outlier_count > 0:
                issues.append(Issue(
                    severity="Medium",
                    category="Outliers",
                    field=col,
                    description=(
                        f"{outlier_count} outliers detected "
                        f"(outside [{lower:.2f}, {upper:.2f}])"
                    ),
                    affected_rows=outlier_count,
                ))

            negative_count = int((df[col] < 0).sum())
            if negative_count > 0:
                issues.append(Issue(
                    severity="Medium",
                    category="Negative Values",
                    field=col,
                    description=f"{negative_count} unexpected negative values",
                    affected_rows=negative_count,
                ))

        # ── Text-specific checks ──────────────────────────
        else:
            str_col = df[col].dropna().astype(str)

            # Inconsistent casing (e.g. "john" vs "JOHN")
            lower_vals = str_col.str.lower()
            if str_col.nunique() > lower_vals.nunique():
                casing_affected = int(
                    str_col[str_col != str_col.str.lower()].shape[0]
                )
                issues.append(Issue(
                    severity="Medium",
                    category="Inconsistent Casing",
                    field=col,
                    description=(
                        "Mixed upper/lower case entries that represent "
                        "the same value (e.g. 'john' vs 'JOHN')"
                    ),
                    affected_rows=casing_affected,
                ))

            # Mixed date formats
            if _has_mixed_date_formats(str_col):
                issues.append(Issue(
                    severity="High",
                    category="Inconsistent Date Format",
                    field=col,
                    description=(
                        "Multiple date formats detected "
                        "(e.g. 01-01-2024 and 15-Mar-24)"
                    ),
                    affected_rows=int(str_col.shape[0]),
                ))

        # ── High cardinality ──────────────────────────────
        unique_ratio = df[col].nunique() / total_rows if total_rows else 0
        if unique_ratio > 0.9 and total_rows > 10:
            issues.append(Issue(
                severity="Low",
                category="High Cardinality",
                field=col,
                description=(
                    f"{df[col].nunique()} unique values "
                    f"({unique_ratio*100:.0f}% of rows)"
                ),
                affected_rows=int(df[col].nunique()),
            ))

    # ── Duplicates (dataset-level) ────────────────────────
    if duplicate_rows > 0:
        issues.append(Issue(
            severity="High",
            category="Duplicates",
            field="Dataset",
            description=f"{duplicate_rows} duplicate rows detected",
            affected_rows=duplicate_rows,
        ))

    return DetectionReport(
        total_rows=total_rows,
        total_columns=total_columns,
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        completeness_pct=completeness_pct,
        issues=issues,
    )
