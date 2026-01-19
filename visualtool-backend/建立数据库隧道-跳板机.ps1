# SSH 隧道脚本（数据库连接，通过跳板机）
# 使用方法：.\建立数据库隧道-跳板机.ps1
# 注意：此窗口需要保持打开，关闭窗口会断开隧道

$localPort = "3307"
$remoteHost = "127.0.0.1"
$remotePort = "3306"
$jumpHost = "z135"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "数据库 SSH 隧道（通过跳板机）" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "跳板机: $jumpHost (10.181.9.135)" -ForegroundColor Yellow
Write-Host "目标服务器: 10.181.142.180:1022" -ForegroundColor Yellow
Write-Host "本地端口: $localPort" -ForegroundColor Yellow
Write-Host "远程 MySQL: $remoteHost:$remotePort" -ForegroundColor Yellow
Write-Host ""
Write-Host "正在建立 SSH 隧道..." -ForegroundColor Green
Write-Host "提示：输入密码后，此窗口需要保持打开" -ForegroundColor Yellow
Write-Host "关闭此窗口将断开隧道连接" -ForegroundColor Red
Write-Host ""
Write-Host "建立成功后，Navicat 连接配置：" -ForegroundColor Cyan
Write-Host "  主机: 127.0.0.1" -ForegroundColor White
Write-Host "  端口: $localPort" -ForegroundColor White
Write-Host "  用户名: root" -ForegroundColor White
Write-Host "  密码: 123456" -ForegroundColor White
Write-Host ""
Write-Host "按 Ctrl+C 可以断开连接" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 建立 SSH 隧道
ssh -J $jumpHost -L ${localPort}:${remoteHost}:${remotePort} -p 1022 ls@10.181.142.180

Write-Host ""
Write-Host "SSH 隧道已断开" -ForegroundColor Red
