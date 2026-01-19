# Navicat SSH 隧道连接 MySQL 指南

## 连接信息说明

通过 SSH 隧道连接远程服务器的 MySQL 数据库，配置如下：

### MySQL 连接信息（远程服务器上的数据库）
- **主机**：`127.0.0.1`（通过 SSH 隧道后访问）
- **端口**：`3306`
- **用户名**：`root`
- **密码**：`123456`

### SSH 隧道信息（用于建立安全连接）
- **SSH 主机**：`10.181.142.180`
- **SSH 端口**：`1022`
- **SSH 用户名**：`ls`
- **SSH 密码**：`camel`

## Navicat 配置步骤

### 步骤 1：打开 Navicat 新建连接

1. 打开 Navicat（MySQL 版本）
2. 点击左上角 **"连接"** 按钮
3. 选择 **"MySQL"**

### 步骤 2：配置常规连接信息

在 **"常规"** 选项卡中填写：

```
连接名：HPC-服务器（可自定义）
主机名或IP地址：127.0.0.1
端口：3306
用户名：root
密码：123456
```

⚠️ **注意**：这里的主机是 `127.0.0.1`，因为通过 SSH 隧道后，远程数据库会映射到本地。

### 步骤 3：配置 SSH 隧道

1. 切换到 **"SSH"** 选项卡
2. 勾选 **"使用SSH隧道"**
3. 填写 SSH 连接信息：

```
SSH主机：10.181.142.180
SSH端口：1022
SSH用户名：ls
SSH密码：camel
```

### 步骤 4：测试连接

1. 点击 **"测试连接"** 按钮
2. 如果配置正确，会显示 **"连接成功"**
3. 点击 **"确定"** 保存连接

### 步骤 5：连接数据库

1. 双击连接名称打开连接
2. 如果提示输入密码，输入 MySQL 密码：`123456`
3. 成功连接后可以看到数据库列表

## 连接示意图

```
本地 Navicat
    ↓
SSH 隧道 (10.181.142.180:1022)
    ↓
远程服务器 (ls@10.181.142.180)
    ↓
MySQL (127.0.0.1:3306, root/123456)
```

## 常见问题排查

### 1. 连接失败：无法建立 SSH 连接

**可能原因**：
- 服务器 IP 或端口错误
- 网络不通（需要 VPN 或内网）
- SSH 服务未运行

**解决方法**：
```powershell
# 先测试 SSH 连接是否正常
ssh -p 1022 ls@10.181.142.180
```

### 2. 连接失败：无法连接到 MySQL

**可能原因**：
- MySQL 服务未启动
- MySQL 只监听 127.0.0.1（需要确认）
- 数据库用户权限问题

**解决方法**（在服务器上执行）：
```bash
# 检查 MySQL 是否运行
sudo systemctl status mysql
# 或
sudo service mysql status

# 启动 MySQL（如果未运行）
sudo systemctl start mysql
# 或
sudo service mysql start

# 检查 MySQL 监听地址
sudo netstat -tuln | grep 3306
# 应该看到 127.0.0.1:3306 或 0.0.0.0:3306
```

### 3. 连接失败：Access denied for user 'root'

**可能原因**：
- 密码错误
- 用户不存在
- 用户没有远程访问权限

**解决方法**（在服务器上执行）：
```bash
# 登录 MySQL
mysql -u root -p

# 检查用户和权限
SELECT user, host FROM mysql.user WHERE user='root';

# 如果需要，创建数据库
CREATE DATABASE IF NOT EXISTS visualtool;

# 检查数据库是否存在
SHOW DATABASES;
```

### 4. SSH 隧道建立成功但无法访问数据库

**可能原因**：
- MySQL 配置只允许本地连接
- 防火墙阻止

**解决方法**：
检查 MySQL 配置文件 `/etc/mysql/mysql.conf.d/mysqld.cnf` 或 `/etc/my.cnf`：
```ini
# 确保 bind-address 是 127.0.0.1（允许本地连接）
bind-address = 127.0.0.1
```

## 使用命令行建立 SSH 隧道（备用方案）

如果 Navicat 的 SSH 隧道有问题，可以手动建立隧道：

### Windows PowerShell

```powershell
# 建立 SSH 隧道（将远程 3306 映射到本地 3307）
ssh -L 3307:127.0.0.1:3306 -p 1022 ls@10.181.142.180

# 保持这个窗口打开，然后在 Navicat 中连接：
# 主机：127.0.0.1
# 端口：3307
```

### 使用 PuTTY（图形界面）

1. 打开 PuTTY
2. 在 **Session** 中填写：
   - Host Name: `10.181.142.180`
   - Port: `1022`
3. 在左侧选择 **Connection > SSH > Tunnels**
4. 添加端口转发：
   - Source port: `3307`
   - Destination: `127.0.0.1:3306`
   - 选择 **Local**
   - 点击 **Add**
5. 回到 **Session**，保存配置，点击 **Open**
6. 登录后保持窗口打开
7. 在 Navicat 中连接 `127.0.0.1:3307`

## 在服务器上初始化数据库

如果数据库还没有创建，需要在服务器上执行：

```bash
# 连接到服务器
ssh -p 1022 ls@10.181.142.180

# 登录 MySQL
mysql -u root -p
# 输入密码：123456

# 创建数据库
CREATE DATABASE IF NOT EXISTS visualtool CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# 创建用户（如果需要）
CREATE USER IF NOT EXISTS 'root'@'localhost' IDENTIFIED BY '123456';
GRANT ALL PRIVILEGES ON visualtool.* TO 'root'@'localhost';
FLUSH PRIVILEGES;

# 退出
EXIT;
```

## 验证连接

连接成功后，你应该能看到：
- `visualtool` 数据库（如果已创建）
- 其他系统数据库（如 `information_schema`, `mysql` 等）

## 安全建议

⚠️ **重要提示**：
1. 生产环境建议使用强密码
2. 考虑创建专用数据库用户（而不是 root）
3. 定期备份数据库
4. 不要在代码中硬编码密码
