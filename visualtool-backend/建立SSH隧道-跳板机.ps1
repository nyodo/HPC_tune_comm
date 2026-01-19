# SSH 隧道脚本（通过跳板机连接服务器）
# 使用方法：.\建立SSH隧道-跳板机.ps1
# 注意：此窗口需要保持打开，关闭窗口会断开隧道

$localPort = "8108"
$remoteHost = "127.0.0.1"
$remotePort = "8108"
$jumpHost = "z135"  # 使用 SSH 配置中的别名
$targetHost = "hpc-server"  # 使用 SSH 配置中的别名

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SSH 隧道连接（通过跳板机）" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "跳板机: $jumpHost (10.181.9.135)" -ForegroundColor Yellow
Write-Host "目标服务器: 10.181.142.180:1022" -ForegroundColor Yellow
Write-Host "本地端口: $localPort" -ForegroundColor Yellow
Write-Host "远程端口: $remotePort" -ForegroundColor Yellow
Write-Host ""
Write-Host "正在建立 SSH 隧道..." -ForegroundColor Green
Write-Host "提示：输入密码后，此窗口需要保持打开" -ForegroundColor Yellow
Write-Host "关闭此窗口将断开隧道连接" -ForegroundColor Red
Write-Host ""
Write-Host "建立成功后，前端可以访问：" -ForegroundColor Cyan
Write-Host "  http://127.0.0.1:8080" -ForegroundColor White
Write-Host ""
Write-Host "前端配置已设置为：target: http://127.0.0.1:8108" -ForegroundColor Gray
Write-Host ""
Write-Host "按 Ctrl+C 可以断开连接" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 SSH 配置是否存在
$sshConfigPath = "$env:USERPROFILE\.ssh\config"
if (-not (Test-Path $sshConfigPath)) {
    Write-Host "警告: SSH 配置文件不存在: $sshConfigPath" -ForegroundColor Yellow
    Write-Host "将使用完整命令连接..." -ForegroundColor Yellow
    Write-Host ""
    # 使用完整命令
    ssh -J zhaoyk@10.181.9.135 -L ${localPort}:${remoteHost}:${remotePort} -p 1022 ls@10.181.142.180
} else {
    # 使用 SSH 配置中的别名（需要先配置）
    # 如果配置了 hpc-server，使用：
    # ssh -L ${localPort}:${remoteHost}:${remotePort} hpc-server
    # 否则使用完整命令：
    Write-Host "使用跳板机连接..." -ForegroundColor Cyan
    ssh -J $jumpHost -L ${localPort}:${remoteHost}:${remotePort} -p 1022 ls@10.181.142.180
}

Write-Host ""
Write-Host "SSH 隧道已断开" -ForegroundColor Red
