# SSH 动态端口转发脚本（用于 MT-3000 连接）
# 使用方法：.\建立MT3000代理隧道.ps1
# 注意：此窗口需要保持打开，关闭窗口会断开隧道

$jumpHost = "z135"
$targetHost = "10.181.142.180"
$targetPort = "1022"
$socksPort = "1080"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MT-3000 SOCKS5 代理隧道" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "跳板机: $jumpHost (10.181.9.135)" -ForegroundColor Yellow
Write-Host "目标服务器: ${targetHost}:${targetPort}" -ForegroundColor Yellow
Write-Host "SOCKS5 代理端口: $socksPort" -ForegroundColor Yellow
Write-Host ""
Write-Host "正在建立 SSH 动态端口转发..." -ForegroundColor Green
Write-Host "提示：此操作会在服务器上创建 SOCKS5 代理" -ForegroundColor Yellow
Write-Host "该代理会通过你的本地 EasyConnect 连接超算" -ForegroundColor Yellow
Write-Host ""
Write-Host "重要：" -ForegroundColor Red
Write-Host "  1. 确保本地 EasyConnect 已连接 XJTU 内网" -ForegroundColor Yellow
Write-Host "  2. 此窗口需要保持打开" -ForegroundColor Yellow
Write-Host "  3. 关闭此窗口会断开隧道" -ForegroundColor Yellow
Write-Host ""
Write-Host "建立成功后，后端可以连接 MT-3000：" -ForegroundColor Cyan
Write-Host "  proxy_host: 127.0.0.1" -ForegroundColor White
Write-Host "  proxy_port: $socksPort" -ForegroundColor White
Write-Host ""
Write-Host "按 Ctrl+C 可以断开连接" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 建立 SSH 动态端口转发
# -D 1080: 在服务器上创建 SOCKS5 代理（端口 1080）
# -N: 不执行远程命令，只建立隧道
# -J: 通过跳板机连接
ssh -J $jumpHost -D $socksPort -p $targetPort ls@$targetHost -N

Write-Host ""
Write-Host "SSH 隧道已断开" -ForegroundColor Red
