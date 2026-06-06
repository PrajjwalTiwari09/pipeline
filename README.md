# 🔬 AI Data Quality Pipeline

> **Hackathon Submission** · Microsoft AI Stack · Data Intelligence Track  
> _"Data is everywhere, but insight is rare."_

---

## 📌 Project Overview

The **AI Data Quality Pipeline** is an end-to-end intelligent data cleaning and insight platform. Upload any messy CSV or Excel file and the pipeline will:

1. **Detect** — automatically scan for missing values, duplicates, outliers, inconsistent casing, mixed date formats, negative values, and high-cardinality columns.
2. **Clean** — remove duplicates, fill nulls (median for numeric, mode for text), normalise casing, and standardise date formats to ISO 8601.
3. **Analyse** — render interactive Plotly charts that visualise issue distribution.
4. **Explain** — call **Azure OpenAI GPT-4o** to generate an executive summary with a data health score, root-cause hypotheses, and actionable recommendations.
5. **Query** — ask natural language questions about your data and get instant answers powered by GPT-4o.
6. **Export** — download the fully cleaned CSV with one click.

---

## 🤖 Microsoft AI Stack Used

| Component | Usage |
|---|---|
| **Azure OpenAI (GPT-4o)** | AI-powered data quality summaries, NL query engine, recommendations |
| **Azure Cognitive Services** | Hosts the OpenAI deployment endpoint |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-org>/ai-data-quality-pipeline.git
cd ai-data-quality-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your Azure OpenAI credentials:

```dotenv
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

> ⚠️ **Never commit `.env` to source control.** The `.gitignore` excludes it by default.  
> ⚠️ **Never hardcode credentials in Python files.** Always use `os.getenv()`.

### 5. Test your Azure connection

```bash
python test_connection.py
```

This will verify your endpoint, API key, and deployment name are all working before you launch the app.

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
.
├── app.py                   # Streamlit frontend & pipeline orchestration
├── Detector.py              # Issue detection engine
├── Cleaner.py               # Data cleaning engine
├── NLQueryEngine.py         # Natural language query engine (GPT-4o)
├── azure_openai_service.py  # Azure OpenAI AI summary integration
├── test_connection.py       # Azure connection verifier
├── requirements.txt         # Python dependencies
├── .env                     # Your credentials (never commit this)
├── .env.example             # Environment variable template
└── .gitignore               # Excludes .env and build artefacts
```

---

## 🔍 Detection Capabilities (`Detector.py`)

The `detect(df)` function scans the entire DataFrame and returns a `DetectionReport` containing all identified issues.

| Check | Severity | Method |
|---|---|---|
| Missing / null values | High | `isnull().sum()` per column |
| Duplicate rows | High | `df.duplicated()` |
| Inconsistent date formats (e.g. `01-01-2024` vs `15-Mar-24`) | High | Regex heuristic on string columns |
| Statistical outliers | Medium | IQR method (1.5× rule) |
| Inconsistent casing (`john` vs `JOHN`) | Medium | `nunique()` vs lowercased `nunique()` |
| Unexpected negative values | Medium | `(df[col] < 0).sum()` |
| High cardinality columns | Low | unique ratio > 90% and > 10 rows |

### Output — `DetectionReport`

```python
@dataclass
class DetectionReport:
    total_rows: int
    total_columns: int
    missing_cells: int
    duplicate_rows: int
    completeness_pct: float
    issues: List[Issue]
```

Each `Issue` contains `severity`, `category`, `field`, `description`, and `affected_rows`.

---

## 🧹 Cleaning Steps (`Cleaner.py`)

The `clean(df)` function returns a cleaned DataFrame and a report dictionary.

| Step | Action |
|---|---|
| 1 | Remove exact duplicate rows |
| 2 | Normalise text casing to title-case (e.g. `JOHN` → `John`) |
| 3 | Standardise date strings to `YYYY-MM-DD` (ISO 8601) |
| 4 | Fill missing numeric values with column **median** |
| 5 | Fill missing text values with column **mode** |
| 6 | Flag (but do not remove) outliers and negative values |

### Cleaning Report Keys

```python
{
    "duplicates_removed": int,
    "nulls_filled": int,
    "casing_normalised": int,
    "dates_standardised": int,
    "cleaned_rows": int,
    "negative_values": [{"column": str, "count": int}],
    "outliers_flagged": [{"column": str, "count": int}],
}
```

---

## 💬 Natural Language Query Engine (`NLQueryEngine.py`)

Ask plain English questions about your data. GPT-4o converts them into safe pandas code, executes it, and returns the result.

### How it works

1. Sends the DataFrame schema, column names, and first 3 rows to GPT-4o.
2. GPT-4o returns a JSON object with `code` and `explanation`.
3. The code is checked against a **safety blocklist** (no `os`, `exec`, `open`, etc.).
4. Code is executed in a sandboxed namespace with whitelisted builtins only.
5. If execution fails, a **one-shot retry** with the error message is attempted automatically.

### Example queries

```
"Show me the top 10 rows"
"How many rows have missing values in the Sales column?"
"What is the average revenue by region?"
"Are there any negative values in the Amount column?"
```

### Safety blocklist

The engine blocks any generated code containing:

```
import os, import sys, import subprocess, __import__,
exec(, eval(, os.system, os.popen, shutil, pathlib,
open(, write(, delete, remove(
```

### QueryResult dataclass

```python
@dataclass
class QueryResult:
    success: bool
    question: str
    pandas_code: str
    result: Any               # DataFrame, scalar, or str
    result_type: str          # "dataframe" | "scalar" | "error"
    explanation: str
    error: str = ""
```

---

## 🤖 AI Summary (`azure_openai_service.py`)

The `generate_ai_summary(data, issues)` function calls GPT-4o with a structured prompt and returns an executive summary containing:

1. **Overall Data Health Score** (0–100) with justification
2. **Top 3 Critical Issues** needing immediate attention
3. **Root Cause Hypotheses** — likely upstream process failures
4. **Actionable Recommendations** — concrete next steps
5. **Hidden Patterns** — interesting correlations or anomalies

---

## 🔐 Security Best Practices

### Never hardcode credentials

```python
# ❌ WRONG
api_key = "abc123yourkeyhere"

# ✅ CORRECT
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
```

### Keep `.env` out of Git

Your `.gitignore` must contain:
```
.env
*.env
```

If you accidentally committed secrets:
```bash
# Untrack the file
git rm --cached .env

# Purge from entire history
git filter-repo --path .env --invert-paths --force
git push origin --force --all
```

> ⚠️ If a key was ever pushed to a remote repo, **rotate it immediately** in the Azure Portal regardless of history cleanup.

### Test your connection safely

```bash
python test_connection.py
```

This script tries multiple API versions and reports which one works — without printing your full key.

---

## 🧪 Sample Data

A minimal CSV to demonstrate the pipeline:

```
Name,Sales,Date
john,1000,01-01-2024
JOHN,1000,02-01-2024
alice,500,10-02-2024
bob,250,15-Mar-24
```

Issues demonstrated: duplicate rows, inconsistent casing, mixed date formats.

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.35.0 | Web UI framework |
| `pandas` | ≥2.0.0 | Data manipulation |
| `numpy` | ≥1.26.0 | Numerical operations |
| `plotly` | ≥5.18.0 | Interactive charts |
| `openai` | ≥1.30.0 | Azure OpenAI SDK |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |
| `pyarrow` | ≥14.0.0 | Parquet file support |
| `openpyxl` | ≥3.1.0 | Excel `.xlsx` support |
| `xlrd` | ≥2.0.1 | Excel `.xls` (legacy) support |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 👥 Team

| Name | Role |
|---|---|
| Learner09 | Full-stack & AI Integration |

---

## 📄 Licence
