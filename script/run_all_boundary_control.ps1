<#
.SYNOPSIS
Honor a process-bound stop before tandem, or apply staged files after that run exits.
.DESCRIPTION
Check is called by the pending tandem batch. It skips work only for the exact
requesting master PID/start time. ApplyAfterExit waits for that master and then
copies reviewed staged files after validating destination/source hashes. Neither
mode interrupts a running fit or launches a numerical workflow.
.PARAMETER Mode
Check returns 10 for the matching one-time skip, otherwise 0. ApplyAfterExit
waits for the recorded master and installs the staged update plan.
.PARAMETER RequestPath
Control JSON path; defaults to state/run_all_control/stop_after_single.json.
#>
[CmdletBinding()]
param(
    [ValidateSet('Check', 'ApplyAfterExit')]
    [string]$Mode = 'Check',
    [string]$RequestPath
)

$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ([string]::IsNullOrWhiteSpace($RequestPath)) {
    $RequestPath = Join-Path $repository 'state\run_all_control\stop_after_single.json'
}
if (-not (Test-Path -LiteralPath $RequestPath)) { exit 0 }
$request = Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json

function Save-ControlState {
    <# .SYNOPSIS
    Persist this request's state and UTC update time without changing its target identity.
    #>
    $request.updated_utc = [DateTime]::UtcNow.ToString('o')
    $request | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $RequestPath -Encoding UTF8
}

function Get-MatchingMaster {
    <# .SYNOPSIS
    Return the recorded master only when both PID and precise UTC start time match.
    #>
    $process = Get-Process -Id $request.master_pid -ErrorAction SilentlyContinue
    if ($null -ne $process -and $process.StartTime.ToUniversalTime().Ticks -eq $request.master_start_utc_ticks) {
        return $process
    }
    return $null
}

function Resolve-WorkspaceFile([string]$RelativePath) {
    <# .SYNOPSIS
    Resolve a staged/source destination and reject paths outside this checkout.
    #>
    $resolved = [IO.Path]::GetFullPath((Join-Path $repository $RelativePath))
    if (-not $resolved.StartsWith($repository + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Control-plan path is outside this checkout: $RelativePath"
    }
    return $resolved
}

if ($Mode -eq 'Check') {
    if ($request.phase -ne 'Requested') { exit 0 }
    $launcher = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
    if ($launcher.ParentProcessId -ne $request.master_pid -or $null -eq (Get-MatchingMaster)) { exit 0 }
    $request.phase = 'StoppedAfterSingle'
    $request.boundary_reached_utc = [DateTime]::UtcNow.ToString('o')
    Save-ControlState
    Write-Output 'Requested stop: single-cylinder experiments finished; tandem experiments are skipped for this run.'
    exit 10
}

try {
    $master = Get-MatchingMaster
    if ($null -ne $master) { $master.WaitForExit() }
    # Reload to retain the checkpoint state written as the tandem call was skipped.
    $request = Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json
    if ($request.phase -notin @('Requested', 'StoppedAfterSingle')) { exit 0 }
    if ($request.files.Count -eq 0) { throw 'No staged update files were registered.' }
    foreach ($file in $request.files) {
        $source = Resolve-WorkspaceFile $file.source
        $destination = Resolve-WorkspaceFile $file.destination
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $file.source_sha256) {
            throw "Staged file changed: $($file.source)"
        }
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $file.expected_destination_sha256) {
            throw "Destination changed while waiting; preserved for review: $($file.destination)"
        }
    }
    $request.phase = 'ApplyingUpdates'
    Save-ControlState
    foreach ($file in $request.files) {
        Copy-Item -LiteralPath (Resolve-WorkspaceFile $file.source) -Destination (Resolve-WorkspaceFile $file.destination) -Force
    }
    $request.phase = 'ReadyForNextRun'
    $request.applied_utc = [DateTime]::UtcNow.ToString('o')
    Save-ControlState
    Write-Output 'The master exited. Staged optimizer experiments and runner counts are installed for the next launch.'
    exit 0
}
catch {
    $request.phase = 'UpdateFailed'
    $request.error = $_.Exception.Message
    Save-ControlState
    throw
}
