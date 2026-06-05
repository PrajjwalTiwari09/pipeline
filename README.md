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
5. **Export** — download the fully cleaned CSV with one click.

---

## 🤖 Microsoft AI Stack Used

| Component | Usage |
|---|---|
| **Azure OpenAI (GPT-4o)** | AI-powered data quality summaries and recommendations |
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
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2025-01-01-preview
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

> ⚠️ **Never commit `.env` to source control.** The `.gitignore` excludes it by default.

### 5. Run the app

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
├── azure_openai_service.py  # Azure OpenAI integration
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Excludes .env and build artefacts
└── sample.csv               # Sample dataset for testing
```

---

## 🔍 Detection Capabilities

| Check | Severity |
|---|---|
| Missing / null values | High |
| Duplicate rows | High |
| Inconsistent date formats (e.g. `01-01-2024` vs `15-Mar-24`) | High |
| Statistical outliers (IQR method) | Medium |
| Inconsistent casing (`john` vs `JOHN`) | Medium |
| Unexpected negative values | Medium |
| High cardinality columns | Low |

---

## 🧹 Cleaning Steps

1. Remove exact duplicate rows.
2. Normalise text casing (title-case).
3. Standardise date strings to `YYYY-MM-DD`.
4. Fill missing numeric values with column median.
5. Fill missing text values with column mode.
6. Flag (but do not remove) outliers and negatives.

---

## 🧪 Sample Data

A minimal `sample.csv` is included to demonstrate the pipeline:

```
Name,Sales,Date
john,1000,01-01-2024
JOHN,1000,02-01-2024      ← duplicate + casing issue
alice,500,10-02-2024
bob,250,15-Mar-24          ← mixed date format
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `openai` | Azure OpenAI SDK |
| `python-dotenv` | Environment variable loading |

---

## 👥 Team

| Name | Role |
|---|---|
| Bhargav | Full-stack & AI Integration |

---

## 📄 Licence

MIT — see `LICENSE` for details.