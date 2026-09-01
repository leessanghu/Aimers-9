$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$venv = Join-Path $PSScriptRoot ".venv-cpu"
$req = Join-Path $PSScriptRoot "requirements-cpu.txt"

Write-Host "Project root: $root"
Write-Host "CPU env: $venv"

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    $version = py -3.11 --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        py -3.11 -m venv $venv
    } else {
        throw "Python 3.11 is required. Install Python 3.11 or create the venv manually."
    }
} else {
    $plain = python --version
    if ($plain -notmatch "3\.11\.") {
        throw "Python 3.11 is required. Current python is: $plain"
    }
    python -m venv $venv
}

$pip = Join-Path $venv "Scripts\python.exe"
& $pip -m pip install --upgrade pip
& $pip -m pip install -r $req
& $pip (Join-Path $PSScriptRoot "check_env.py") --track cpu

Write-Host "Done. Activate with:"
Write-Host "  .\envs\latest_tabular\.venv-cpu\Scripts\Activate.ps1"

