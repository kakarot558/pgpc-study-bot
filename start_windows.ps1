# StudyBuddy launcher for Windows PowerShell.
# If Windows says the script is not digitally signed, run it like this instead:
#   powershell -ExecutionPolicy Bypass -File .\start_windows.ps1
# Creates the virtual environment on first run, installs packages, builds the model, starts the app.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path ".venv")) {
    Write-Host "First run: creating virtual environment..."
    py -3 -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}
& .\.venv\Scripts\python.exe -m ml.train
$port = if ($env:STUDYBUDDY_PORT) { $env:STUDYBUDDY_PORT } else { "5000" }
Write-Host "Starting... open http://127.0.0.1:$port in your browser. The first start can take up to a minute; wait, then press Ctrl+C once to stop."
& .\.venv\Scripts\python.exe app.py
