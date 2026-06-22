$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$WslPython = Join-Path $Root ".venv_wsl\bin\python"
$WindowsPython = Join-Path $Root ".venv\Scripts\python.exe"
$App = "scripts/final_demo_history_app.py"

if ((Get-Command wsl.exe -ErrorAction SilentlyContinue) -and (Test-Path $WslPython)) {
    $WslRoot = (wsl.exe wslpath -a $Root).Trim()
    Write-Host "Starting demo in WSL environment: .venv_wsl"
    wsl.exe bash -lc "cd '$WslRoot' && .venv_wsl/bin/python -m streamlit run $App"
    exit $LASTEXITCODE
}

if (Test-Path $WindowsPython) {
    Write-Host "WSL environment not found. Starting with .venv"
    & $WindowsPython -m streamlit run $App
    exit $LASTEXITCODE
}

Write-Host "Virtual environment not found. Starting with python from PATH"
python -m streamlit run $App
exit $LASTEXITCODE
