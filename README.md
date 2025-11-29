# LLM Analysis Quiz - Endpoint

This FastAPI project runs a webhook endpoint `/quiz-webhook` that accepts quiz tasks, visits a provided quiz URL using Playwright, extracts data and answers, optionally calls an LLM (via OpenAI or AIPipe), and submits the answer back.

## Quick local run

1. Create a virtualenv and activate it:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
