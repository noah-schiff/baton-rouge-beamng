param([string]$OutDir = "C:\Dev\BeamNG-SELA\docs\screenshots", [string]$Prefix = "",
      [int]$Settle = 8, [int]$TimeoutSec = 600)
# Optional: brmapTest.lua takes its own engine screenshots (<userfolder>/screenshots/brmap/),
# which other windows can't spoil. This saves full-screen captures besides them: it polls
# <userfolder>/temp/brmapTest_status.txt (written by brmapTest.lua) and saves a screen
# capture <Settle> seconds after each "view <name>" line. Stops at "done", which comes
# after the traffic check that follows the views.
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$status = Join-Path $env:LOCALAPPDATA "BeamNG\BeamNG.drive\current\temp\brmapTest_status.txt"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$last = ""
$start = Get-Date
while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSec) {
    $line = (Get-Content $status -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($line -and $line -ne $last) {
        $last = $line
        $msg = ($line -split " ", 2)[1]
        if ($msg -eq "done") { "done"; exit 0 }
        if ($msg -like "view *") {
            $name = $msg.Substring(5)
            Start-Sleep -Seconds $Settle
            $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
            $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
            $g = [System.Drawing.Graphics]::FromImage($bmp)
            $g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
            $path = Join-Path $OutDir "$Prefix$name.png"
            $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
            $g.Dispose(); $bmp.Dispose()
            "saved $path"
        }
    }
    Start-Sleep -Milliseconds 300
}
"timed out"
