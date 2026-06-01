<#
.SYNOPSIS
  One-command autonomous launch: data -> detached training -> live dashboard.

.DESCRIPTION
  Kicks the whole thing off and returns. The trainer and dashboard run as
  independent background processes, so you can close this window and training
  continues. The trainer auto-downloads enwik8 if --Corpus is missing, holds out
  a validation set, saves the best checkpoint, decays LR on plateaus, recovers
  from divergence, and resumes from its last checkpoint on restart.

.EXAMPLE
  .\start.ps1
  .\start.ps1 -Corpus mydata.txt -Port 9000 -MaxSteps 500000
#>
param(
  [string]$Corpus  = "enwik8",
  [int]   $Port    = 8000,
  [string]$CkptDir = "checkpoints",
  [int]   $MaxSteps = 200000,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found at $py - create the 3.12 ROCm venv first (see README)." }

# 1. Make the package importable from anywhere (no PYTHONPATH needed).
& $py -c "import thcm" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "[setup] installing thcm (editable) ..."
  & $py -m pip install -e . | Out-Null
}

# 2. Preflight - fail loudly if the GPU isn't really active.
Write-Host "[preflight]"
& $py src\thcm\utils\device.py

New-Item -ItemType Directory -Force $CkptDir | Out-Null

# 3. Launch the trainer detached (it auto-downloads enwik8 if missing).
$trainArgs = @('-m','thcm.training.auto','--corpus',$Corpus,'--resume',
               '--ckpt-dir',$CkptDir,'--max-steps',$MaxSteps)
$train = Start-Process -PassThru -WindowStyle Hidden -FilePath $py -ArgumentList $trainArgs `
  -RedirectStandardOutput "$CkptDir\console.out.log" -RedirectStandardError "$CkptDir\console.err.log"
Write-Host "[train]     PID $($train.Id)  (logs: $CkptDir\training.log, console.*.log)"

# 4. Launch the dashboard detached.
$dashArgs = @('-m','thcm.training.dashboard','--ckpt-dir',$CkptDir,'--port',$Port)
$dash = Start-Process -PassThru -WindowStyle Hidden -FilePath $py -ArgumentList $dashArgs `
  -RedirectStandardOutput "$CkptDir\dashboard.log" -RedirectStandardError "$CkptDir\dashboard.err.log"
Write-Host "[dashboard] PID $($dash.Id)  ->  http://localhost:$Port"

# 5. Record PIDs so stop.ps1 can shut them down.
"$($train.Id)`n$($dash.Id)" | Out-File "$CkptDir\pids.txt" -Encoding ascii

if (-not $NoBrowser) { Start-Sleep -Seconds 2; Start-Process "http://localhost:$Port" }

Write-Host ""
Write-Host "Running autonomously. Close this window anytime - training continues."
Write-Host "Watch it:  http://localhost:$Port"
Write-Host "Stop it :  .\stop.ps1"
