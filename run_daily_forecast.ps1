# Runs the research collection cycle under the current Windows user.
# Configure with Task Scheduler; no credentials are embedded in this file.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot 'work\convlstm-env\Scripts\python.exe'
$logDirectory = Join-Path $projectRoot 'work\monitoring\daily-forecast'
$statusFile = Join-Path $projectRoot 'outputs\glo12-daily-cycle-latest.json'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'The project Python environment is unavailable.'
}
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$runDate = [DateTime]::UtcNow.ToString('yyyy-MM-dd')
$logFile = Join-Path $logDirectory ("cycle-$stamp.log")
$errorLogFile = Join-Path $logDirectory ("cycle-$stamp.err.log")
Push-Location -LiteralPath $projectRoot
try {
    $arguments = '-m ml.run_daily_forecast_cycle --date ' + $runDate + ' --status-output "' + $statusFile + '"'
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $projectRoot -RedirectStandardOutput $logFile `
        -RedirectStandardError $errorLogFile -WindowStyle Hidden -Wait -PassThru
    $runExit = $process.ExitCode
    if ($runExit -ne 0) {
        [Console]::Error.WriteLine("Forecast collection failed (exit $runExit). See $errorLogFile")
        exit $runExit
    }
    $status = Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json
    if ($status.observation_capture_pending_dates.Count -gt 0 -or
        $status.pending_observations.Count -gt 0 -or
        $status.remaining_backlog -gt 0) {
        [Console]::Error.WriteLine("Forecast collection has pending sources. See $statusFile and $errorLogFile")
        exit 2
    }
    exit 0
}
finally {
    Pop-Location
}
