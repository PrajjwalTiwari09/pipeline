"""
test_connection.py
Run this to verify your Azure OpenAI connection is working.
Usage: python test_connection.py
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# ── Step 1: Load .env ─────────────────────────────────────
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
loaded = load_dotenv(dotenv_path=dotenv_path)

print("=" * 50)
print("AZURE OPENAI CONNECTION TEST")
print("=" * 50)

# ── Step 2: Check .env was found ──────────────────────────
print(f"\n[1] .env file found     : {'YES' if loaded else 'NO  ← .env missing or empty'}")

# ── Step 3: Check each variable ───────────────────────────
endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key    = os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

print(f"[2] ENDPOINT loaded     : {'YES → ' + endpoint[:40] + '...' if endpoint else 'NO  ← missing in .env'}")
print(f"[3] KEY loaded          : {'YES → ' + api_key[:10] + '...' if api_key else 'NO  ← missing in .env'}")
print(f"[4] DEPLOYMENT loaded   : {'YES → ' + deployment if deployment else 'NO  ← missing in .env (default: gpt-4o)'}")

# ── Step 4: Stop early if any variable is missing ─────────
if not endpoint or not api_key:
    print("\n❌ FAILED: Fix the missing values above in your .env file first.")
    exit(1)

# ── Step 5: Try multiple api_versions ────────────────────
print("\n[5] Connecting to Azure OpenAI...")

api_versions = [
    "2025-01-01-preview",
    "2024-11-20",
    "2024-08-01-preview",
    "2024-02-01",
]

response = None
working_version = None

for version in api_versions:
    try:
        print(f"   Trying api_version: {version} ...")
        client = AzureOpenAI(
            api_key=api_key,
            api_version=version,
            azure_endpoint=endpoint,
        )
        response = client.chat.completions.create(
            model=deployment or "gpt-4o",
            messages=[
                {"role": "user", "content": "Reply with exactly: CONNECTION OK"}
            ],
            max_tokens=10,
            temperature=0,
        )
        working_version = version
        break
    except Exception as ve:
        print(f"   ✗ Failed: {ve}")

if response:
    reply = response.choices[0].message.content.strip()
    print(f"\n[6] Model replied       : {reply}")
    print(f"[7] Working api_version : {working_version}")
    print("\n✅ SUCCESS: Azure OpenAI is connected and working!")
    print(f"\n   ➜ Use api_version='{working_version}' in azure_openai_service.py")
else:
    print("\n❌ FAILED: All api_versions tried, none worked.")
    print("   Double-check your endpoint and key in Azure Portal.")

print("=" * 50)
