# run_and_test.ps1 - local runner + test POST
cd (Split-Path -Parent $MyInvocation.MyCommand.Path)

# activate venv
if (-Not (Test-Path ".venv")) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Playwright browsers (only first run)
python -m playwright install --with-deps chromium

# set ephemeral env vars for this shell
$env:SECRET = "ankit1212"
$env:EMAIL = "23f1001763@ds.study.iitm.ac.in"
$env:OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDE3NjNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.ZBrMvwiHRHgoB6fwC_xZPeJAlt_cmGIqb3A9p0dPTTw"
$env:OPENAI_BASE_URL = "https://aipipe.org/openai/v1"

# start server in foreground
Write-Host "Starting server on 127.0.0.1:3000 - ctrl+c to stop"
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m uvicorn main:app --host 127.0.0.1 --port 3000"
Start-Sleep -Seconds 3

# send demo POST (optional)
Invoke-RestMethod -Uri "http://127.0.0.1:3000/quiz-webhook" -Method POST -Headers @{ "Content-Type" = "application/json" } -Body '{"email":"23f1001763@ds.study.iitm.ac.in","secret":"ankit1212","url":"https://tds-llm-analysis.s-anand.net/demo?email=23f1001763@ds.study.iitm.ac.in"}'
Write-Host "demo POST sent. Check server logs in the opened process."
