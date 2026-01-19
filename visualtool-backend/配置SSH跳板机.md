# 配置 SSH 跳板机连接

## 步骤 1：编辑 SSH 配置文件

打开或创建文件：`C:\Users\nyodo\.ssh\config`

添加以下配置：

```
# 跳板机配置（已有）
Host z135
    User zhaoyk
    HostName 10.181.9.135
    Port 22
    IdentityFile "C:\Users\nyodo\.ssh\id_rsa_zhaoyk"

# 目标服务器配置（通过跳板机）
Host hpc-server
    User ls
    HostName 10.181.142.180
    Port 1022
    ProxyJump z135
    # 如果需要密码认证
    PreferredAuthentications password
```

## 步骤 2：测试连接

```powershell
# 现在可以直接使用别名连接
ssh hpc-server

# 或建立隧道
ssh -L 8108:127.0.0.1:8108 hpc-server
```

## 步骤 3：使用便捷脚本

运行 `.\建立SSH隧道-跳板机.ps1` 即可自动建立隧道。
