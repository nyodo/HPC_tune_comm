# PowerShell 脚本：快速连接到服务器
# 使用方法：.\快速连接服务器.ps1

$serverIP = "10.181.142.180"
$serverPort = "1022"
$username = "ls"

Write-Host "正在连接到服务器..." -ForegroundColor Green
Write-Host "服务器: $serverIP:$serverPort" -ForegroundColor Yellow
Write-Host "用户名: $username" -ForegroundColor Yellow
Write-Host "密码: camel" -ForegroundColor Yellow
Write-Host ""

# 使用 SSH 连接
ssh -p $serverPort $username@$serverIP
