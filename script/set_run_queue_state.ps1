<#
.SYNOPSIS
Pause or resume the verified process tree belonging to an existing run queue.
.DESCRIPTION
Reads status.json and checks the helper's command line and process identity.
Pause records each suspended process in pause.json; Resume checks recorded PID
and UTC start time before restoring it. Workers are resumed before their parent
runner/helper, and pause failures attempt to resume already suspended workers.
.PARAMETER Action
Pause an active queue or Resume the processes listed in its paused manifest.
.PARAMETER QueueDirectory
Existing queue output directory containing status.json and pause/resume state.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Pause', 'Resume')]
    [string]$Action,
    [Parameter(Mandatory = $true)]
    [string]$QueueDirectory
)

$ErrorActionPreference = 'Stop'
$QueueDirectory = (Resolve-Path -LiteralPath $QueueDirectory).Path
$statusPath = Join-Path $QueueDirectory 'status.json'
$pausePath = Join-Path $QueueDirectory 'pause.json'
$status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class RunQueueProcessControl {
    [DllImport("ntdll.dll")]
    public static extern int NtSuspendProcess(IntPtr handle);
    [DllImport("ntdll.dll")]
    public static extern int NtResumeProcess(IntPtr handle);
}
'@

function Save-PauseRecord {
    <#
    .SYNOPSIS
    Write the enclosing pause/resume manifest to this queue's pause.json.
    .DESCRIPTION
    Called after each process transition so an interrupted operation retains
    the recorded process identities and their individual resume state.
    #>
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $pausePath -Encoding UTF8
}

function Resume-RecordedProcess($entry) {
    <#
    .SYNOPSIS
    Resume a still-matching recorded process and persist its resume timestamp.
    .DESCRIPTION
    Verify PID and start-time ticks before calling NtResumeProcess. An exited
    or replaced PID is not resumed. Native errors propagate; successful or no
    longer needed entries receive a UTC timestamp in the pause manifest.
    #>
    if ($entry.resumed_utc) { return }
    $process = Get-Process -Id $entry.pid -ErrorAction SilentlyContinue
    if ($null -ne $process -and $process.StartTime.ToUniversalTime().Ticks -eq $entry.start_time_utc_ticks) {
        $result = [RunQueueProcessControl]::NtResumeProcess($process.Handle)
        if ($result -ne 0) { throw "Could not resume process $($entry.pid): NTSTATUS $($result.ToString('X8'))." }
    }
    $entry.resumed_utc = [DateTime]::UtcNow.ToString('o')
    Save-PauseRecord
}

if ($Action -eq 'Pause') {
    if ($status.phase -eq 'Paused') { throw 'This queue is already paused.' }
    if ($status.phase -notin @('Waiting', 'Ready', 'Running')) { throw "Queue is $($status.phase), not active." }
    $root = Get-CimInstance Win32_Process -Filter "ProcessId = $($status.helper_pid)"
    if ($null -eq $root -or $root.Name -ne 'powershell.exe' -or
        $root.CommandLine -notlike '*run_data_diagnostics_after_process.ps1*' -or
        $root.CommandLine.IndexOf($QueueDirectory, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw 'The queue helper identity could not be verified; no processes were suspended.'
    }
    $record = [ordered]@{
        phase = 'Pausing'; previous_phase = $status.phase
        paused_utc = [DateTime]::UtcNow.ToString('o'); processes = @()
    }
    Save-PauseRecord
    $pending = @($root)
    try {
        while ($pending.Count -gt 0) {
            foreach ($info in $pending) {
                $process = Get-Process -Id $info.ProcessId -ErrorAction SilentlyContinue
                if ($null -eq $process) { continue }
                if ([Math]::Abs(($process.StartTime.ToUniversalTime() - $info.CreationDate.ToUniversalTime()).TotalMilliseconds) -gt 1) {
                    throw "Process $($info.ProcessId) changed identity during pause."
                }
                $entry = [ordered]@{
                    pid = $process.Id; parent_pid = $info.ParentProcessId; name = $info.Name
                    start_time_utc_ticks = $process.StartTime.ToUniversalTime().Ticks
                    command_line = $info.CommandLine; resumed_utc = $null
                }
                $result = [RunQueueProcessControl]::NtSuspendProcess($process.Handle)
                if ($result -ne 0) { throw "Could not suspend process $($entry.pid): NTSTATUS $($result.ToString('X8'))." }
                $record.processes += $entry
                Save-PauseRecord
            }
            # Freeze parents first, then rescan so newly launched workers cannot escape.
            $frozenIds = @($record.processes | ForEach-Object { $_.pid })
            $pending = @(Get-CimInstance Win32_Process | Where-Object {
                $_.ParentProcessId -in $frozenIds -and $_.ProcessId -notin $frozenIds -and $_.Name -ne 'conhost.exe'
            })
        }
        $record.phase = 'Paused'
        Save-PauseRecord
    }
    catch {
        $pauseError = $_
        for ($index = $record.processes.Count-1; $index -ge 0; $index--) {
            Resume-RecordedProcess $record.processes[$index]
        }
        $record.phase = 'PauseFailed'
        Save-PauseRecord
        throw $pauseError
    }
    $status.phase = 'Paused'
}
else {
    $record = Get-Content -LiteralPath $pausePath -Raw | ConvertFrom-Json
    if ($record.phase -ne 'Paused') { throw "Pause record is $($record.phase); refusing to resume twice." }
    # Resume workers before their parent runner and queue helper.
    for ($index = $record.processes.Count-1; $index -ge 0; $index--) {
        Resume-RecordedProcess $record.processes[$index]
    }
    $record.phase = 'Resumed'
    $record | Add-Member -NotePropertyName resumed_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
    Save-PauseRecord
    $status.phase = $record.previous_phase
}
$status.updated_utc = [DateTime]::UtcNow.ToString('o')
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
Add-Content -LiteralPath (Join-Path $QueueDirectory 'queue.log') -Encoding UTF8 `
    -Value ($status.updated_utc + ' ' + $Action + ': ' + $record.processes.Count + ' queue/worker processes.')
[ordered]@{ phase = $status.phase; process_ids = @($record.processes | ForEach-Object { $_.pid }) } | ConvertTo-Json
