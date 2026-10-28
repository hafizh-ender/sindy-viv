<#
.SYNOPSIS
Wait for one recorded process identity and then run this checkout's queued batches.
.DESCRIPTION
Matches both PID and UTC start-time ticks before waiting. Runs run_all.bat by
default, or the batches in a JSON plan, captures stdout/stderr, and maintains status.json and queue.log
in OutputDirectory. The process identity prevents a reused PID from blocking
the queue for an unrelated process.
.PARAMETER WaitProcessId
PID of the existing process that must finish before this checkout is launched.
.PARAMETER WaitProcessStartTimeUtcTicks
UTC start-time ticks recorded with that PID; both fields identify the process.
.PARAMETER OutputDirectory
Directory containing queue status, event log, and master-batch output logs.
.PARAMETER DiagnosticsDryRun
Forward --dry-run to queued batches instead of computing diagnostics or models.
.PARAMETER BatchPlanPath
Optional JSON with a batches array and optional readiness_state/required_phase.
The readiness JSON is checked after waiting and before launching any batch.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$WaitProcessId,
    [Parameter(Mandatory = $true)]
    [long]$WaitProcessStartTimeUtcTicks,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [switch]$DiagnosticsDryRun,
    [string]$BatchPlanPath
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$batches = @('run_all.bat')
$batchPlan = $null
if ($BatchPlanPath) {
    $batchPlan = Get-Content -LiteralPath $BatchPlanPath -Raw | ConvertFrom-Json
    $batches = @($batchPlan.batches)
}
$null = New-Item -ItemType Directory -Path $OutputDirectory -Force
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$statusPath = Join-Path $OutputDirectory 'status.json'
$eventLog = Join-Path $OutputDirectory 'queue.log'
$status = [ordered]@{
    phase = 'Starting'
    helper_pid = $PID
    waiting_for_pid = $WaitProcessId
    waiting_for_start_time_utc_ticks = $WaitProcessStartTimeUtcTicks
    repository = $repository
    diagnostics_dry_run = [bool]$DiagnosticsDryRun
    batches = $batches
    batch_plan = $BatchPlanPath
    updated_utc = $null
    active_batch = $null
    active_pid = $null
    runs = @()
    error = $null
}

function Write-QueueStatus([string]$Phase, [string]$Message) {
    <#
    .SYNOPSIS
    Persist the current queue phase and append a UTC-stamped event message.
    .DESCRIPTION
    Updates the enclosing status mapping, replaces status.json, and appends to
    queue.log. Both paths and the helper/process identity belong to this queue.
    #>
    $status.phase = $Phase
    $status.updated_utc = [DateTime]::UtcNow.ToString('o')
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding UTF8
    Add-Content -LiteralPath $eventLog -Encoding UTF8 -Value ($status.updated_utc + ' ' + $Message)
}

try {
    if ($batches.Count -eq 0) { throw 'The queue plan contains no batches.' }
    foreach ($batch in $batches) {
        if ([IO.Path]::GetFileName($batch) -ne $batch -or [IO.Path]::GetExtension($batch) -ne '.bat') {
            throw "Queued batches must be .bat filenames in this checkout: $batch"
        }
        if (-not (Test-Path -LiteralPath (Join-Path $repository $batch) -PathType Leaf)) {
            throw "Required batch file is missing: $batch"
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $repository '.venv\Scripts\python.exe') -PathType Leaf)) {
        throw 'The project Python environment is missing.'
    }

    # The start time prevents a reused PID from delaying the queue for another process.
    $watched = Get-Process -Id $WaitProcessId -ErrorAction SilentlyContinue
    if ($null -ne $watched -and -not $watched.HasExited -and
        $watched.StartTime.ToUniversalTime().Ticks -eq $WaitProcessStartTimeUtcTicks) {
        Write-QueueStatus 'Waiting' "Waiting for the original batch process $WaitProcessId to exit."
        $watched.WaitForExit()
    }
    if ($null -ne $batchPlan -and $batchPlan.readiness_state) {
        $ready = Get-Content -LiteralPath (Join-Path $repository $batchPlan.readiness_state) -Raw | ConvertFrom-Json
        if ($ready.phase -ne $batchPlan.required_phase) {
            throw "Queued updates are not ready: phase=$($ready.phase); expected=$($batchPlan.required_phase)."
        }
    }
    Write-QueueStatus 'Ready' 'The prerequisite process exited and readiness checks passed; starting the queued batches.'

    $failed = $false
    foreach ($batch in $batches) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($batch)
        $stdout = Join-Path $OutputDirectory ($stem + '.stdout.log')
        $stderr = Join-Path $OutputDirectory ($stem + '.stderr.log')
        $arguments = @('/d', '/c', $batch)
        if ($DiagnosticsDryRun) { $arguments += '--dry-run' }
        $status.active_batch = $batch
        Write-QueueStatus 'Running' "Starting $batch."
        $started = [DateTime]::UtcNow.ToString('o')
        $child = Start-Process -FilePath $env:ComSpec -ArgumentList $arguments -WorkingDirectory $repository `
            -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        # Retain the process handle so Windows PowerShell preserves ExitCode after exit.
        $null = $child.Handle
        $status.active_pid = $child.Id
        Write-QueueStatus 'Running' "$batch is running as process $($child.Id)."
        $child.WaitForExit()
        $exitCode = $child.ExitCode
        $status.runs += [ordered]@{
            batch = $batch
            pid = $child.Id
            started_utc = $started
            finished_utc = [DateTime]::UtcNow.ToString('o')
            exit_code = $exitCode
            stdout_log = $stdout
            stderr_log = $stderr
        }
        if ($null -eq $exitCode -or $exitCode -ne 0) { $failed = $true }
        $status.active_pid = $null
        Write-QueueStatus 'Running' "$batch finished with exit code $exitCode."
        $child.Dispose()
    }
    $status.active_batch = $null
    if ($failed) {
        Write-QueueStatus 'FinishedWithErrors' 'The queued workflows finished; inspect the logs for failures.'
        exit 1
    }
    Write-QueueStatus 'Finished' 'All queued workflows finished successfully.'
    exit 0
}
catch {
    $status.error = $_.Exception.Message
    Write-QueueStatus 'Failed' $status.error
    exit 1
}
