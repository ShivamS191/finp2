# main.py
import os
import re
import json
import base64
import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright, Page
import httpx
import pandas as pd
from PyPDF2 import PdfReader

# --- Configuration and logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-quiz-endpoint")

SECRET = os.getenv("SECRET")
DEFAULT_EMAIL = os.getenv("EMAIL", "you@example.com")
TIME_LIMIT_SECONDS = int(os.getenv("TIME_LIMIT_SECONDS", 3 * 60))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # must be set to enable LLM calls
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
USE_LLM = bool(OPENAI_API_KEY)

app = FastAPI(title="LLM Analysis Quiz Endpoint")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str
    class Config:
        extra = "allow"

# ---------------- LLM helper ----------------
async def call_llm_chat(messages: list, model: str = "gpt-4o-mini") -> str:
    """
    Async call to OpenAI-like Chat Completions via OPENAI_BASE_URL.
    Uses OPENAI_API_KEY as the Bearer token.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 800,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        j = r.json()
        if isinstance(j.get("choices"), list) and j["choices"]:
            msg = j["choices"][0].get("message", {}).get("content")
            return msg if msg is not None else ""
        # fallback
        return j.get("result", "") or json.dumps(j)

# ---------------- Utility helpers ----------------
def try_extract_and_decode_base64(html: str) -> Optional[str]:
    m = re.search(r'atob\(\s*`([^`]*)`', html, re.S)
    if not m:
        m = re.search(r'atob\(\s*\'([^\']*)\'', html, re.S)
    if not m:
        m = re.search(r'atob\(\s*\"([^\"]*)\"', html, re.S)
    if not m:
        return None
    b64 = m.group(1).strip()
    try:
        decoded = base64.b64decode(b64).decode("utf8", errors="ignore")
        logger.info("Decoded base64 payload from page")
        return decoded
    except Exception as e:
        logger.warning("Failed to base64-decode payload: %s", str(e))
        return None

def strip_html_to_text(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_embedded_json(html: str) -> Optional[Any]:
    m = re.search(r"<pre[^>]*>(\{[\s\S]*?\})</pre>", html, re.I)
    if not m:
        m = re.search(r"(\{[\s\S]{20,3000}\})", html, re.I)
    if not m:
        return None
    blob = m.group(1)
    try:
        return json.loads(blob)
    except Exception:
        blob_fixed = re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", blob))
        try:
            return json.loads(blob_fixed)
        except Exception:
            return None

def find_submit_url_in_page(html: str, text: str) -> Optional[str]:
    links = re.findall(r'https?://[^\s\'"<>]+', html)
    for u in links:
        if "submit" in u.lower():
            return u
    tlinks = re.findall(r'https?://[^\s\'"<>]+', text)
    for u in tlinks:
        if "submit" in u.lower():
            return u
    return None

# ---------------- Resource parsing (CSV, XLSX, PDF) ----------------
async def fetch_and_extract_answer_from_resource(page: Page, resource_url: str, deadline: float) -> Optional[Any]:
    logger.info("Downloading resource %s", resource_url)
    try:
        timeout_ms = int(max(5000, min(20000, (deadline - asyncio.get_event_loop().time()) * 1000)))
        resp = await page.request.get(resource_url, timeout=timeout_ms)
        content_type = resp.headers.get("content-type", "")
        raw = await resp.body()
    except Exception as e:
        logger.warning("Failed to download resource: %s", str(e))
        return None

    # CSV
    if resource_url.lower().endswith(".csv") or "text/csv" in content_type:
        try:
            text = raw.decode("utf8", errors="ignore")
            from io import StringIO
            df = pd.read_csv(StringIO(text))
            for col in df.columns:
                if "value" in str(col).lower():
                    return float(pd.to_numeric(df[col], errors="coerce").sum(skipna=True))
            numeric_cols = df.select_dtypes(include="number")
            if not numeric_cols.empty:
                return float(numeric_cols.sum().sum())
        except Exception:
            text = raw.decode("utf8", errors="ignore")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if len(lines) < 2:
                return None
            headers = [h.strip() for h in lines[0].split(",")]
            try:
                idx = next(i for i,h in enumerate(headers) if "value" in h.lower())
            except StopIteration:
                idx = None
            if idx is None:
                return None
            s = 0.0
            for row in lines[1:]:
                cols = row.split(",")
                if len(cols) > idx:
                    rawv = re.sub(r"[^0-9\.\-eE+]", "", cols[idx])
                    try:
                        s += float(rawv)
                    except Exception:
                        pass
            return s

    # xlsx/xls
    if resource_url.lower().endswith((".xlsx", ".xls")) or "spreadsheet" in content_type or "excel" in content_type:
        try:
            import io
            df = pd.read_excel(io.BytesIO(raw))
            for col in df.columns:
                if "value" in str(col).lower():
                    return float(pd.to_numeric(df[col], errors="coerce").sum(skipna=True))
            numeric_cols = df.select_dtypes(include="number")
            if not numeric_cols.empty:
                return float(numeric_cols.sum().sum())
        except Exception as e:
            logger.warning("Failed to parse xlsx: %s", e)
        return None

    # pdf
    if resource_url.lower().endswith(".pdf") or "application/pdf" in content_type:
        try:
            from io import BytesIO
            reader = PdfReader(BytesIO(raw))
            full_text = []
            for p in reader.pages:
                try:
                    t = p.extract_text() or ""
                    full_text.append(t)
                except Exception:
                    pass
            txt = "\n".join(full_text)
            m = re.search(r"sum of .*?value.*?([0-9][0-9,.\s]*)", txt, re.I | re.S)
            if m:
                num = re.sub(r"[^\d\.\-eE]", "", m.group(1))
                try:
                    return float(num)
                except Exception:
                    pass
            m2 = re.search(r"([0-9]+(?:[.,][0-9]+)+)", txt)
            if m2:
                return float(m2.group(1).replace(",", ""))
        except Exception as e:
            logger.warning("PDF parse failed: %s", str(e))
        return None

    # fallback try to parse numeric content
    try:
        text = raw.decode("utf8", errors="ignore")
        m = re.search(r"([0-9]+(?:[.,][0-9]+)*)", text)
        if m:
            return float(m.group(1).replace(",", ""))
    except Exception:
        pass

    return None

# ---------------- Core processing ----------------
@app.post("/quiz-webhook")
async def quiz_webhook(payload: QuizRequest, background_tasks: BackgroundTasks):
    arrival = asyncio.get_event_loop().time()
    if not SECRET:
        raise HTTPException(status_code=500, detail="Server SECRET not configured")
    if payload.secret != SECRET:
        logger.warning("Invalid secret for %s", payload.email)
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(process_quiz_chain, payload.dict(), arrival)
    return JSONResponse(status_code=200, content={"status": "accepted"})

async def process_quiz_chain(payload: Dict[str, Any], arrival_time: float):
    deadline = arrival_time + TIME_LIMIT_SECONDS
    async def time_left() -> float:
        return max(0.0, deadline - asyncio.get_event_loop().time())

    logger.info("Starting quiz processing for email=%s url=%s", payload.get("email"), payload.get("url"))
    if await time_left() <= 0:
        logger.warning("No time left to process payload at start")
        return

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            context = await browser.new_context()
            page = await context.new_page()

            current_url = payload.get("url")
            last_submit_response = None
            tries = 0

            while current_url and (await time_left()) > 3 and tries < 30:
                tries += 1
                tl = await time_left()
                logger.info("Visiting %s (time left %.1fs)", current_url, tl)

                page_target = current_url
                if not current_url.startswith("http://") and not current_url.startswith("https://") and not current_url.startswith("file://"):
                    page_target = "file://" + current_url

                try:
                    await page.goto(page_target, wait_until="networkidle", timeout=int(min(30000, tl * 1000)))
                except Exception as e:
                    logger.error("page.goto failed for %s: %s", page_target, str(e))
                    break

                page_html = await page.content()
                page_text = await page.evaluate("() => document.documentElement.innerText || ''")

                decoded_html = try_extract_and_decode_base64(page_html)
                if decoded_html:
                    page_html = decoded_html
                    page_text = strip_html_to_text(decoded_html)

                embedded_json = extract_embedded_json(page_html)
                submit_url = None
                if isinstance(embedded_json, dict):
                    submit_url = embedded_json.get("submit") or embedded_json.get("submit_url") or embedded_json.get("submitUrl")
                if not submit_url:
                    submit_url = find_submit_url_in_page(page_html, page_text)
                if not submit_url:
                    logger.error("No submit URL found on page %s", current_url)
                    break

                answer = None
                resource_url = None
                if isinstance(embedded_json, dict):
                    if "answer" in embedded_json:
                        answer = embedded_json["answer"]
                    resource_url = embedded_json.get("url") or embedded_json.get("resource")

                if not resource_url:
                    try:
                        links = await page.eval_on_selector_all("a", "els => els.map(e => e.href).filter(Boolean)")
                        for u in links:
                            if u.lower().endswith(('.csv', '.xlsx', '.xls', '.pdf')):
                                resource_url = u
                                break
                    except Exception:
                        pass

                if resource_url and (await time_left()) > 5:
                    answer = await fetch_and_extract_answer_from_resource(page, resource_url, await time_left())

                if answer is None:
                    if USE_LLM and re.search(r"sum of|what is the sum|calculate", page_text, re.I):
                        prompt = (
                            "You are a helpful data-analysis assistant. Given the page text below, "
                            "extract a clear instruction of what to compute and, if possible, compute an answer or explain what resource to fetch.\n\n"
                            "Page text:\n\n" + page_text[:6000]
                        )
                        try:
                            llm_out = await call_llm_chat([{"role":"user","content":prompt}])
                            logger.info("LLM output (truncated): %s", (llm_out or "")[:400])
                            m = re.search(r"(-?\d+(?:[.,]\d+)*)", llm_out or "")
                            if m:
                                answer = float(m.group(1).replace(",", ""))
                            else:
                                answer = (llm_out or "").strip()[:800]
                        except Exception as e:
                            logger.warning("LLM call failed: %s", e)
                            answer = None

                if answer is None:
                    nums = re.findall(r"([+-]?\d+(?:[,\.]\d+)*)", page_text)
                    if nums:
                        try:
                            numbers = [float(n.replace(",", "")) for n in nums]
                            answer = sum(numbers)
                        except Exception:
                            answer = page_text.strip().splitlines()[0][:800]
                    else:
                        answer = page_text.strip().splitlines()[0][:800]

                submit_payload = {
                    "email": payload.get("email") or DEFAULT_EMAIL,
                    "secret": SECRET,
                    "url": current_url,
                    "answer": answer
                }

                logger.info("Submitting to %s payload keys: %s", submit_url, list(submit_payload.keys()))
                try:
                    tl = int(min(20000, (await time_left()) * 1000))
                    submit_resp = await page.request.post(submit_url, data=submit_payload, timeout=tl)
                    try:
                        submit_body = await submit_resp.json()
                    except Exception:
                        submit_body = None
                    logger.info("Submit response status %s body %s", submit_resp.status, submit_body)
                    last_submit_response = submit_body or {"status": submit_resp.status}
                    if isinstance(submit_body, dict) and submit_body.get("url"):
                        current_url = submit_body.get("url")
                        continue
                    else:
                        break
                except Exception as e:
                    logger.error("Error submitting to %s : %s", submit_url, str(e))
                    break

            logger.info("Processing finished. last_submit_response=%s", last_submit_response)
            try:
                await context.close()
                await browser.close()
            except Exception:
                pass

    except Exception as exc:
        logger.exception("Unexpected error in process_quiz_chain: %s", str(exc))
    finally:
        logger.info("process_quiz_chain ended for email=%s", payload.get("email"))

@app.get("/llm-test")
async def llm_test():
    """Simple endpoint to verify LLM connectivity."""
    if not OPENAI_API_KEY:
        return {"ok": False, "reason": "OPENAI_API_KEY not set"}
    try:
        out = await call_llm_chat([{"role": "user", "content": "What is 2 + 2?"}], model="gpt-4o-mini")
        return {"ok": True, "response": out}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.utcnow().isoformat()}

@app.get("/")
async def root():
    return {"message": "LLM Quiz Analyzer API running"}
