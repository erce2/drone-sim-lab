param(
    [string]$Bat = "C:\temp\lab3.bat",
    [string]$Name = "lab3",
    [int]$TimeoutSec = 900,
    [string[]]$Filter = @("LAB3", "Traceback", "Error")
)
# 清掉旧日志, 避免读到上一轮的结果 (踩过一次)
ssh gpu "cmd /c del /q C:\temp\jobs\$Name.log" 2>$null | Out-Null
ssh gpu "cmd /c powershell -NoProfile -ExecutionPolicy Bypass -File C:\temp\run_job.ps1 -Name $Name -BatPath $Bat" 2>&1 |
    Select-String "gpu_used|started"
$deadline = (Get-Date).AddSeconds($TimeoutSec)
Start-Sleep -Seconds 5
while ((Get-Date) -lt $deadline) {
    $t = ssh gpu "cmd /c type C:\temp\jobs\$Name.log" 2>$null | Out-String
    if ($t -match 'EXITCODE=') {
        $t -split "`n" | Select-String ($Filter -join "|")
        return
    }
    Start-Sleep -Seconds 8
}
Write-Output "=== TIMEOUT ==="
ssh gpu "cmd /c type C:\temp\jobs\$Name.log" 2>$null | Select-Object -Last 40