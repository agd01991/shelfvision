$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$WslPython = Join-Path $Root ".venv_wsl\bin\python"
$WindowsPython = Join-Path $Root ".venv\Scripts\python.exe"
$Script = "scripts/defense_demo_smoke_check.py"
$ForwardedArgs = @($args)

if ((Get-Command wsl.exe -ErrorAction SilentlyContinue) -and (Test-Path $WslPython)) {
    $WslRoot = (wsl.exe wslpath -a $Root).Trim()
    $QuotedArgs = ($ForwardedArgs | ForEach-Object { "'" + ($_ -replace "'", "'\"'\"'") + "'" }) -join " "
    Write-Host "Running smoke check in WSL environment: .venv_wsl"
    wsl.exe bash -lc "cd '$WslRoot' && .venv_wsl/bin/python $Script $QuotedArgs"
    exit $LASTEXITCODE
}

if (Test-Path $WindowsPython) {
    Write-Host "WSL environment not found. Running smoke check with .venv"
    & $WindowsPython $Script @ForwardedArgs
    exit $LASTEXITCODE
}

Write-Host "Virtual environment not found. Running smoke check with python from PATH"
python $Script @ForwardedArgs
exit $LASTEXITCODE
