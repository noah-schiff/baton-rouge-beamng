param([string]$OutDir = "C:\Dev\BeamNG-SELA\.superpowers\drive_shots", [int]$Every = 20,
      [int]$TimeoutSec = 1800)
# While brmapDrive.lua runs, saves a screen capture every <Every> seconds (named after the
# capture time and the latest status line, listed in index.txt). Stops at "done".
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$status = Join-Path $env:LOCALAPPDATA "BeamNG\BeamNG.drive\current\temp\brmapDrive_status.txt"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$start = Get-Date
$i = 0
while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSec) {
    $line = (Get-Content $status -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($line -and ($line -split " ", 2)[1] -eq "done") { "done"; exit 0 }
    if ($line -and $line -match "route|park|steering") {
        $i++
        $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
        $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
        $path = Join-Path $OutDir ("shot_{0:D3}.png" -f $i)
        $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
        $g.Dispose(); $bmp.Dispose()
        Add-Content (Join-Path $OutDir "index.txt") ("{0:D3} {1:HH:mm:ss} {2}" -f $i, (Get-Date), $line)
    }
    Start-Sleep -Seconds $Every
}
"timed out"
