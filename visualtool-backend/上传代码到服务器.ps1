# PowerShell 脚本：上传代码到服务器
# 使用方法：.\上传代码到服务器.ps1

$serverIP = "10.181.142.180"
$serverPort = "1022"
$username = "ls"
$remotePath = "~/visualtool-backend"
$localPath = "."

Write-Host "正在上传代码到服务器..." -ForegroundColor Green
Write-Host "本地路径: $localPath" -ForegroundColor Yellow
Write-Host "服务器路径: $remotePath" -ForegroundColor Yellow
Write-Host ""

# 使用 SCP 上传（需要输入密码：camel）
scp -P $serverPort -r $localPath $username@${serverIP}:$remotePath

Write-Host ""
Write-Host "上传完成！" -ForegroundColor Green
Write-Host "现在可以连接到服务器进行部署：" -ForegroundColor Yellow
Write-Host "ssh -p $serverPort $username@$serverIP" -ForegroundColor Cyan
