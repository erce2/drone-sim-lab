param(
    [string]$Log = "lab1",
    [int]$TimeoutSec = 1200
)
# 轮询远程日志, 直到出现 EXITCODE= 或超时, 然后把整个日志打出来
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    $txt = ssh gpu "cmd /c type C:\temp\jobs\$Log.log" 2>$null | Out-String
    if ($txt -match 'EXITCODE=') {
        Write-Output $txt
        return
    }
    Start-Sleep -Seconds 10
}
Write-Output "=== TIMEOUT waiting for $Log ==="
ssh gpu "cmd /c type C:\temp\jobs\$Log.log" 2>$null