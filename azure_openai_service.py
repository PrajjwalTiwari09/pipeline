"""
azure_openai_service.py
Handles all Azure OpenAI communication.
Credentials are loaded from environment variables — never hardcoded.
"""

import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

def _get_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key  = os.getenv("AZURE_OPENAI_KEY")

    if not endpoint or not api_key:
        raise EnvironmentError(
            "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_KEY. "
            "Please set them in your .env file."
        )

    return AzureOpenAI(
        api_key=api_key,
        api_version="2025-01-01-preview",
        azure_endpoint=endpoint,
    )


def generate_ai_summary(data: str, issues: str) -> str:
    """
    Generate AI-powered data quality insights from cleaned data
    and the detected issue summary.
    """
    client     = _get_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    prompt = f"""You are a senior data quality analyst. 
Given the dataset sample and detected issues below, produce a concise executive summary with:

1. **Overall Data Health Score** (0-100) with a brief justification.
2. **Top 3 Critical Issues** that need immediate attention.
3. **Root Cause Hypotheses** — what upstream process likely caused these issues?
4. **Actionable Recommendations** — concrete next steps the data team should take.
5. **Hidden Patterns** — any interesting correlations or anomalies worth investigating further.

---
DATASET SAMPLE:
{data}

---
DETECTED ISSUES:
{issues}
---

Keep your response structured, specific, and business-focused. Avoid generic advice."""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class data quality expert. "
                    "You provide precise, actionable insights from raw data."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=900,
    )

    return response.choices[0].message.content
