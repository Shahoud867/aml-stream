# One-time setup of the GPU environment on a Windows machine. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File gpu_session\setup_windows.ps1
# Rehearsal on a machine without a GPU (CPU build of torch):  ... -File gpu_session\setup_windows.ps1 -Cpu -Venv C:\temp\venv-rehearsal
param(
    [string]$Venv = ".venv-gpu",
    [string]$Official = "..\aml_official",
    [switch]$Cpu
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Step($m) { Write-Host "`n=== $m" -ForegroundColor Cyan }
function Run($exe, $argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $exe $($argv -join ' ')" }
}

Step "Python"
$py = $null
foreach ($v in "3.11", "3.10", "3.12") {
    try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break } } catch { }
}
if (-not $py) {
    try {
        $ver = & python -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($ver -in "3.10", "3.11", "3.12") { $py = @("python") }
    } catch { }
}
if (-not $py) { throw "Need Python 3.10, 3.11 or 3.12 (torch 2.5.1 has no wheels for newer versions). Install 3.11 from python.org." }
Write-Host ("using: " + ($py -join " "))

Step "Virtual environment: $Venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    $rest = @(); if ($py.Length -gt 1) { $rest = $py[1..($py.Length - 1)] }
    Run $py[0] ($rest + @("-m", "venv", $Venv))
}
$vpy = (Resolve-Path "$Venv\Scripts\python.exe").Path
Run $vpy @("-m", "pip", "install", "-q", "--upgrade", "pip")

$flavour = if ($Cpu) { "cpu" } else { "cu121" }
Step "torch 2.5.1 ($flavour). The default pip build on Windows has NO GPU support, so the index is given explicitly"
Run $vpy @("-m", "pip", "install", "-q", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/$flavour")

Step "PyG extensions matching torch 2.5.1+$flavour (binary wheels only; a source build would need a compiler)"
Run $vpy @("-m", "pip", "install", "-q", "--only-binary", ":all:", "torch-scatter", "torch-sparse", "-f", "https://data.pyg.org/whl/torch-2.5.1+$flavour.html")

Step "Everything else"
Run $vpy @("-m", "pip", "install", "-q", "torch-geometric==2.7.0", "wandb", "tqdm", "numpy<2", "pandas", "-r", "requirements-kaggle.txt")

Step "Official Multi-GNN and FraudGT at the pinned commits (cloned from the local submodules, so no internet is needed)"
New-Item -ItemType Directory -Force $Official | Out-Null
$official = (Resolve-Path $Official).Path
$pins = @{ "Multi-GNN" = "252b0252afca109d1d216c411c59ff70753b25fc"; "FraudGT" = "f88db9dc003a6f521a90631c4741541686043119" }
foreach ($name in $pins.Keys) {
    $target = Join-Path $official $name
    $local = Join-Path $root "third_party\$name"
    if (-not (Test-Path "$target\.git")) {
        if (Test-Path "$local\.git") { Run git @("clone", "-q", $local, $target) }
        else { Run git @("clone", "-q", "https://github.com/" + $(if ($name -eq "Multi-GNN") { "IBM/Multi-GNN" } else { "junhongmit/FraudGT" }) + ".git", $target) }
    }
    Run git @("-C", $target, "checkout", "-q", $pins[$name])
    $head = (& git -C $target rev-parse HEAD).Trim()
    if ($head -ne $pins[$name]) { throw "$name is at $head, expected $($pins[$name])" }
    Write-Host "$name at $head"
}

Step "Check"
Run $vpy @("gpu_session\check_env.py", "--official", $official) + $(if ($Cpu) { @("--allow-cpu") } else { @() })
Write-Host "`nSetup done. Environment: $vpy" -ForegroundColor Green
