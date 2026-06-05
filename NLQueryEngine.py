"""
NLQueryEngine.py
Converts natural language questions into pandas operations
using Azure OpenAI GPT-4o, then executes them safely.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from dataclasses import dataclass
from typing import Any

load_dotenv()


# ──────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────

@dataclass
class QueryResult:
    success: bool
    question: str
    pandas_code: str
    result: Any
    result_type: str          # "dataframe" | "scalar" | "error"
    explanation: str
    error: str = ""


# ──────────────────────────────────────────────
# SAFETY CHECK
# ──────────────────────────────────────────────

_BLOCKED = [
    "import os", "import sys", "import subprocess",
    "__import__", "exec(", "eval(",
    "os.system", "os.popen", "shutil", "pathlib",
    "open(", "write(", "delete", "remove(",
]

def _is_safe(code: str) -> bool:
    lowered = code.lower()
    return not any(b in lowered for b in _BLOCKED)


# ──────────────────────────────────────────────
# SAFE EXECUTOR
# Fixed: pass real builtins so pandas works internally
# ──────────────────────────────────────────────

def _execute(code: str, df: pd.DataFrame) -> Any:
    """Execute generated pandas code with a clean but functional namespace."""
    import builtins

    # Whitelist of safe builtins pandas needs internally
    safe_builtins = {
        k: getattr(builtins, k) for k in [
            "len", "range", "enumerate", "zip", "map", "filter",
            "list", "dict", "set", "tuple", "str", "int", "float",
            "bool", "type", "isinstance", "issubclass", "hasattr",
            "getattr", "abs", "round", "min", "max", "sum", "sorted",
            "reversed", "any", "all", "print", "repr", "hash",
            "iter", "next", "vars", "dir", "id", "hex", "oct", "bin",
            "chr", "ord", "format", "divmod", "pow", "callable",
            "NotImplemented", "True", "False", "None",
            "ValueError", "TypeError", "KeyError", "IndexError",
            "AttributeError", "Exception", "StopIteration",
        ]
    }

    namespace = {
        "__builtins__": safe_builtins,
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }

    exec(compile(code, "<query>", "exec"), namespace)
    return namespace.get("result", None)


# ──────────────────────────────────────────────
# SCHEMA BUILDER
# ──────────────────────────────────────────────

def _build_schema(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        dtype  = str(df[col].dtype)
        nulls  = int(df[col].isnull().sum())
        unique = int(df[col].nunique())
        if pd.api.types.is_numeric_dtype(df[col]):
            mn  = round(float(df[col].min(skipna=True)), 2)
            mx  = round(float(df[col].max(skipna=True)), 2)
            avg = round(float(df[col].mean(skipna=True)), 2)
            lines.append(
                f"  - {col} ({dtype}): min={mn}, max={mx}, mean={avg}, "
                f"nulls={nulls}, unique_count={unique}"
            )
        else:
            samples = df[col].dropna().astype(str).unique()[:4].tolist()
            lines.append(
                f"  - {col} ({dtype}): sample_values={samples}, "
                f"nulls={nulls}, unique_count={unique}"
            )
    return "\n".join(lines)


# ──────────────────────────────────────────────
# JSON RESPONSE PARSER
# Handles GPT-4o returning markdown fences or raw JSON
# ──────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    # Strip ```json ... ``` or ``` ... ``` fences
    clean = re.sub(r"^```[a-z]*\s*\n?", "", raw.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\n?```$", "", clean.strip())
    clean = clean.strip()
    return json.loads(clean)


# ──────────────────────────────────────────────
# MAIN ENGINE
# ──────────────────────────────────────────────

class NLQueryEngine:

    SYSTEM_PROMPT = """You are a Python/pandas expert. Convert natural language questions into safe, executable pandas code.

STRICT RULES:
1. The DataFrame is ALWAYS called `df`. Never rename it.
2. ALWAYS assign the final output to a variable named `result`.
3. `result` must be a DataFrame, int, float, or str — nothing else.
4. Never use import statements inside your code.
5. Never use os, sys, open, exec, eval, subprocess.
6. For "show me N rows" → result = df.head(N)
7. For filters → result = df[condition]
8. For counts → result = int(df[condition].shape[0])  or  result = int(df['col'].nunique())
9. For aggregations → result = df.groupby('col')['val'].sum().reset_index()
10. For totals/averages → result = round(float(df['col'].sum()), 2)
11. For yes/no → result = "Yes — N rows found" or "No matches found"
12. Always use .copy() if modifying df.
13. Column names ARE case-sensitive — use exact names from the schema.

RESPONSE FORMAT — respond ONLY with this exact JSON structure, no markdown, no explanation outside JSON:
{"code": "result = df.head(10)", "explanation": "Returns the first 10 rows of the dataset."}"""

    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key  = os.getenv("AZURE_OPENAI_KEY")
        if not endpoint or not api_key:
            raise EnvironmentError(
                "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_KEY in .env"
            )
        self._client = AzureOpenAI(
            api_key=api_key,
            api_version="2025-01-01-preview",
            azure_endpoint=endpoint,
        )
        self._deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    def query(self, question: str, df: pd.DataFrame) -> QueryResult:

        schema   = _build_schema(df)
        col_list = ", ".join(f'"{c}"' for c in df.columns)

        user_msg = (
            f"DataFrame has {len(df)} rows and {len(df.columns)} columns.\n"
            f"Column names (exact, case-sensitive): {col_list}\n\n"
            f"Schema:\n{schema}\n\n"
            f"First 3 rows:\n{df.head(3).to_string(index=False)}\n\n"
            f"Question: {question}"
        )

        # ── Call GPT-4o ───────────────────────────────
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            return QueryResult(
                success=False, question=question,
                pandas_code="", result=None,
                result_type="error", explanation="",
                error=f"GPT-4o API call failed: {e}",
            )

        # ── Parse JSON ────────────────────────────────
        try:
            parsed      = _parse_response(raw)
            code        = parsed.get("code", "").strip()
            explanation = parsed.get("explanation", "")
        except Exception:
            # Last resort: try to extract a code line manually
            code_match = re.search(r'result\s*=.+', raw)
            if code_match:
                code        = code_match.group(0).strip()
                explanation = "Code extracted from response."
            else:
                return QueryResult(
                    success=False, question=question,
                    pandas_code=raw, result=None,
                    result_type="error", explanation="",
                    error=f"Could not parse GPT-4o response:\n{raw}",
                )

        if not code:
            return QueryResult(
                success=False, question=question,
                pandas_code="", result=None,
                result_type="error", explanation="",
                error="GPT-4o returned empty code.",
            )

        # ── Safety check ──────────────────────────────
        if not _is_safe(code):
            return QueryResult(
                success=False, question=question,
                pandas_code=code, result=None,
                result_type="error", explanation="",
                error="Generated code failed safety check.",
            )

        # ── Execute ───────────────────────────────────
        try:
            result = _execute(code, df)
        except Exception as e:
            # Retry once with an error-correction prompt
            try:
                fix_response = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[
                        {"role": "system",    "content": self.SYSTEM_PROMPT},
                        {"role": "user",      "content": user_msg},
                        {"role": "assistant", "content": raw},
                        {"role": "user",      "content":
                            f"That code raised: {e}\n"
                            f"Fix it. Column names are: {col_list}\n"
                            "Return corrected JSON only."},
                    ],
                    temperature=0,
                    max_tokens=400,
                )
                raw2    = fix_response.choices[0].message.content.strip()
                parsed2 = _parse_response(raw2)
                code        = parsed2.get("code", code).strip()
                explanation = parsed2.get("explanation", explanation)
                result      = _execute(code, df)
            except Exception as e2:
                return QueryResult(
                    success=False, question=question,
                    pandas_code=code, result=None,
                    result_type="error", explanation=explanation,
                    error=f"Execution failed: {e}\nRetry also failed: {e2}",
                )

        # ── Classify result ───────────────────────────
        if result is None:
            return QueryResult(
                success=False, question=question,
                pandas_code=code, result=None,
                result_type="error", explanation=explanation,
                error="Code ran but produced no `result`. Make sure your code assigns to `result`.",
            )

        if isinstance(result, pd.DataFrame):
            result_type = "dataframe"
        elif isinstance(result, (int, float, np.integer, np.floating)):
            result_type = "scalar"
            result = round(float(result), 4) if isinstance(result, float) else result
        else:
            result_type = "scalar"
            result = str(result)

        return QueryResult(
            success=True,
            question=question,
            pandas_code=code,
            result=result,
            result_type=result_type,
            explanation=explanation,
        )