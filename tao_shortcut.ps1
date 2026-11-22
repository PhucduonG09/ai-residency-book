# Tao bieu tuong "Residency Book" tren Desktop (chay mot lan).
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop 'Residency Book.lnk'
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath = Join-Path $env:WINDIR 'System32\wscript.exe'
$lnk.Arguments = "`"$(Join-Path $root 'mo_app.vbs')`""
$lnk.WorkingDirectory = $root
$lnk.Description = 'Residency Book - so tay luyen thi AI Residency'
$ico = Join-Path $root 'app\icon.ico'
if (Test-Path $ico) { $lnk.IconLocation = $ico } else { $lnk.IconLocation = (Join-Path $env:WINDIR 'System32\shell32.dll') + ',220' }
$lnk.Save()
Write-Host "Da tao: $lnkPath"
