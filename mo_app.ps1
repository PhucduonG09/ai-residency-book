# Mo app Residency Book nhu mot ung dung rieng (cua so khong thanh dia chi).
# Khoi dong Streamlit an, doi san sang, mo cua so app; dong cua so thi tat server.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8765
$url  = "http://localhost:$port"

function Test-Port {
  try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1', $port); $c.Close(); return $true } catch { return $false }
}
function Fail($msg) {
  [System.Windows.Forms.MessageBox]::Show($msg, 'Residency Book') | Out-Null
  exit 1
}

$server = $null
if (-not (Test-Port)) {
  $stArgs = @('-m','streamlit','run','app\app.py','--server.port',"$port",'--server.headless','true','--browser.gatherUsageStats','false')
  $roots = @("$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3", "$env:LOCALAPPDATA\anaconda3", "$env:LOCALAPPDATA\miniconda3", "C:\ProgramData\anaconda3", "C:\ProgramData\miniconda3")
  $py = $null
  foreach ($r in $roots) {
    $cand = Join-Path $r 'envs\residency-book\python.exe'
    if (Test-Path $cand) { $py = $cand; break }
  }
  if ($py) {
    $server = Start-Process -FilePath $py -ArgumentList $stArgs -WorkingDirectory $root -WindowStyle Hidden -PassThru
  } else {
    $conda = $null
    foreach ($r in $roots) {
      $cand = Join-Path $r 'condabin\conda.bat'
      if (Test-Path $cand) { $conda = $cand; break }
    }
    if (-not $conda) { Fail 'Khong tim thay moi truong residency-book. Hay chay setup.bat truoc.' }
    $cmd = "call `"$conda`" activate residency-book && python -m streamlit run app\app.py --server.port $port --server.headless true --browser.gatherUsageStats false"
    $server = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $cmd -WorkingDirectory $root -WindowStyle Hidden -PassThru
  }
  $ok = $false
  for ($i = 0; $i -lt 120; $i++) {
    if (Test-Port) { $ok = $true; break }
    if ($server.HasExited) { break }
    Start-Sleep -Milliseconds 500
  }
  if (-not $ok) { Fail 'App khong khoi dong duoc. Hay chay run_app.bat de xem loi.' }
}

$browsers = @(
  "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
  "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
  "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
  "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)
$exe = $browsers | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) {
  Start-Process $url
  exit 0
}
# Profile rieng de cua so la mot tien trinh doc lap, de biet khi nao nguoi dung dong app.
$profile = Join-Path $env:LOCALAPPDATA 'ResidencyBookApp'
$win = Start-Process -FilePath $exe -ArgumentList "--app=$url", "--user-data-dir=`"$profile`"", '--no-first-run', '--window-size=1400,900' -PassThru
$win.WaitForExit()
Start-Sleep -Seconds 2
# Dong het cua so con lai dung profile nay (Edge/Chrome co the tach tien trinh)
Get-CimInstance Win32_Process -Filter "Name='msedge.exe' OR Name='chrome.exe'" |
  Where-Object { $_.CommandLine -like '*ResidencyBookApp*' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }
if ($server) {
  # Tat ca cay tien trinh (Streamlit, JupyterLab, kernel) bang taskkill /T
  & taskkill.exe /PID $server.Id /T /F | Out-Null
}
