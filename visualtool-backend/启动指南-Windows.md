# Windows 后端启动指南

## 问题：conda 命令无法识别

在 Windows PowerShell 中，conda 可能未初始化。有以下几种解决方案：

## 方案 1：初始化 Conda（推荐）

### 步骤 1：检查 Conda 是否已安装

```powershell
# 检查 Anaconda/Miniconda 安装路径
# 通常在以下位置之一：
# C:\Users\你的用户名\anaconda3
# C:\Users\你的用户名\miniconda3
# C:\ProgramData\anaconda3
```

### 步骤 2：初始化 Conda for PowerShell

```powershell
# 找到 Anaconda 安装目录后，运行初始化脚本
# 替换路径为你的实际安装路径
& "C:\ProgramData\anaconda3\Scripts\conda.exe" init powershell

# 或者如果安装在用户目录
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" init powershell
```

### 步骤 3：重启 PowerShell 或重新加载配置

```powershell
# 重新加载 PowerShell 配置
. $PROFILE

# 或者直接重启 PowerShell 窗口
```

### 步骤 4：创建环境

```powershell
cd visualtool-backend
conda env create -f environment.yml
conda activate accessmemory  # 注意：environment.yml 中环境名是 accessmemory，不是 LS
```

## 方案 2：使用 Anaconda Prompt（最简单）

1. 打开 **Anaconda Prompt**（从开始菜单搜索）
2. 在 Anaconda Prompt 中运行：

```bash
cd visualtool-backend
conda env create -f environment.yml
conda activate accessmemory
python app.py
```

## 方案 3：使用 Python venv（无需 Conda）

如果不想使用 Conda，可以使用 Python 自带的 venv：

```powershell
cd visualtool-backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 如果遇到执行策略错误，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 安装依赖
pip install -r requirements.txt

# 注意：mysqlclient 在 Windows 上可能需要额外步骤
# 如果安装失败，可以尝试：
# pip install mysqlclient-binary
# 或者使用 pymysql：
# pip install pymysql
# 然后修改 app.py 中的数据库连接字符串为：
# SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:123456@127.0.0.1:3306/visualtool"
```

## 启动后端

```powershell
# 确保已激活环境（conda 或 venv）
python app.py
```

后端将在 `http://127.0.0.1:8108` 启动

## 数据库配置

确保 MySQL 已安装并运行，数据库配置在 `app.py` 中：

```python
SQLALCHEMY_DATABASE_URI = "mysql://root:123456@127.0.0.1:3306/visualtool"
```

首次运行时会自动创建管理员账号：
- 用户名：`admin`
- 密码：`admin`

## 常见问题

### 1. mysqlclient 安装失败（Windows）

**解决方案 A：使用预编译版本**
```powershell
pip install mysqlclient-binary
```

**解决方案 B：使用 pymysql**
```powershell
pip install pymysql
```
然后修改 `app.py` 第 41 行：
```python
SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:123456@127.0.0.1:3306/visualtool"
```

### 2. 端口被占用

如果 8108 端口被占用，修改 `app.py` 最后一行：
```python
app.run(host='127.0.0.1', port=8008, debug=True)  # 改为 8008
```
同时修改前端 `vis/vite.config.js` 中的代理端口。
