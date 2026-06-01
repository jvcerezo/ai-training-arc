<#
.SYNOPSIS
  Stop the background training + dashboard started by start.ps1.
#>
param([string]$CkptDir = "checkpoints")

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root "$CkptDir\pids.txt"
if (-not (Test-Path $pidFile)) { Write-Host "no pids.txt in $CkptDir - nothing to stop."; return }

foreach ($line in Get-Content $pidFile) {
  $procId = $line.Trim()
  if ($procId) {
    Stop-Process -Id ([int]$procId) -Force
    Write-Host "stopped PID $procId"
  }
}
Remove-Item $pidFile
Write-Host "done. (latest model is $CkptDir\best.pt; re-run .\start.ps1 to resume)"
