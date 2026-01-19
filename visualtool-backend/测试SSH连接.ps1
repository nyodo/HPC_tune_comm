# PowerShell 脚本：测试 SSH 连接
# 使用方法：.\测试SSH连接.ps1

$serverIP = "10.181.142.180"
$serverPort = "1022"
$username = "ls"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SSH 连接测试" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "正在测试连接到服务器..." -ForegroundColor Yellow
Write-Host "服务器: $serverIP:$serverPort" -ForegroundColor White
Write-Host "用户名: $username" -ForegroundColor White
Write-Host "密码: camel" -ForegroundColor White
Write-Host ""
Write-Host "如果连接成功，你会看到服务器提示符" -ForegroundColor Cyan
Write-Host "输入 'exit' 可以退出" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 测试 SSH 连接
ssh -p $serverPort $username@$serverIP

Write-Host ""
Write-Host "连接已断开" -ForegroundColor Yellow
