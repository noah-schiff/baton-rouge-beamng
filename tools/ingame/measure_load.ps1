param([Parameter(Mandatory = $true)][string]$Level,
      [string]$Steam = "C:\Program Files (x86)\Steam\steam.exe",
      [int]$SampleSec = 2, [int]$TimeoutSec = 900)
# Measures one load of a brmap level: starts BeamNG.drive through Steam with
# -level <Level> -onLevelLoad_ext brmapTest, then records
#   load_s   the seconds from the launch until <userfolder>\temp\brmapTest_status.txt first
#            changes (brmapTest.lua writes it as soon as the world is ready), and
#   peak_gb  the peak working set of BeamNG.drive.x64.exe, sampled every <SampleSec> s
#            until then (GB of 2^30 bytes, as Task Manager counts),
# and appends "date,level,load_s,peak_gb" to data\logs\loads.csv in the repo. It copies
# the repo's brmapTest.lua into the user folder's lua\ge\extensions\ first (an unpacked mod
# is mounted too late for -onLevelLoad_ext). The game keeps running afterwards, so
# brmapTest goes on with its views; close the game before the next run (one at a time).
# The first load compiles the level's meshes into the user folder's temp\levels\<Level>;
# delete that folder to measure a first load again.
$ErrorActionPreference = "Stop"
$inv = [Globalization.CultureInfo]::InvariantCulture
$exe = "BeamNG.drive.x64"
$user = Join-Path $env:LOCALAPPDATA "BeamNG\BeamNG.drive\current"
$status = Join-Path $user "temp\brmapTest_status.txt"
$csv = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")) "data\logs\loads.csv"

if (Get-Process -Name $exe -ErrorAction SilentlyContinue) {
    Write-Error "BeamNG.drive is already running; close it first (one instance at a time)"
}
$extensions = Join-Path $user "lua\ge\extensions"
New-Item -ItemType Directory -Force $extensions | Out-Null
Copy-Item (Join-Path $PSScriptRoot "brmapTest.lua") $extensions -Force
$before = if (Test-Path $status) { (Get-Item $status).LastWriteTimeUtc } else { $null }

function Sample([ref]$peak) {
    foreach ($p in @(Get-Process -Name $exe -ErrorAction SilentlyContinue)) {
        if ($p.WorkingSet64 -gt $peak.Value) { $peak.Value = $p.WorkingSet64 }
    }
}

$start = Get-Date
Start-Process -FilePath $Steam -ArgumentList "-applaunch", "284160", "-level", $Level, "-onLevelLoad_ext", "brmapTest"
"launched $Level at $($start.ToString('HH:mm:ss')); waiting for $status"
$peak = [long]0
$next = $start
$loaded = $null
while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSec) {
    if ((Get-Date) -ge $next) {
        Sample ([ref]$peak)
        $next = $next.AddSeconds($SampleSec)
    }
    if ((Test-Path $status) -and ($null -eq $before -or (Get-Item $status).LastWriteTimeUtc -ne $before)) {
        $loaded = Get-Date
        break
    }
    Start-Sleep -Milliseconds 250
}
if ($null -eq $loaded) {
    Write-Error "no change to $status within $TimeoutSec s; nothing written to $csv"
}
Sample ([ref]$peak)
$loadS = ($loaded - $start).TotalSeconds.ToString("F1", $inv)
$peakGb = ($peak / 1GB).ToString("F2", $inv)
if (-not (Test-Path $csv)) {
    New-Item -ItemType Directory -Force (Split-Path $csv) | Out-Null
    Set-Content -Path $csv -Value "date,level,load_s,peak_gb" -Encoding ascii
}
$row = "{0},{1},{2},{3}" -f $start.ToString("yyyy-MM-ddTHH:mm:ss", $inv), $Level, $loadS, $peakGb
Add-Content -Path $csv -Value $row -Encoding ascii
"$Level loaded in $loadS s, peak working set $peakGb GB (appended to $csv)"
"the status file reads: $(Get-Content $status -First 1)"
