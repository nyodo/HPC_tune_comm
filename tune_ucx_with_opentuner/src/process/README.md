# 数据处理模块 (Process Module)

## 📋 功能说明

将原始通信日志数据（log-*.csv）转换为统计汇总数据，用于 OpenTuner 调优。

## 🔄 处理流程

```
原始数据 (log-*.csv)
    ↓
1. 读取所有 log-*.csv 文件
    ↓
2. 计算 total_size = sendsize × sendcount
    ↓
3. 按 comm_type 和 total_size 分组统计 count
    ↓
4. 在每个 comm_type 内按 total_size 排序，添加 index
    ↓
处理后数据 (processed.csv)
```

## 📊 数据格式

### 输入数据（log-*.csv）

| 列名 | 说明 |
|------|------|
| comm_type | 通信类型 |
| sendsize | 单次发送大小 |
| sendcount | 发送次数 |
| ... | 其他列 |

### 输出数据（processed.csv）

| 列名 | 说明 | 示例 |
|------|------|------|
| comm_type | 通信类型 | 51 |
| index | 在该 comm_type 下的序号（从1开始） | 1, 2, 3... |
| total_size | 总大小（sendsize × sendcount） | 1200.0 |
| count | 该组合出现的次数 | 1472 |

## 🚀 使用方法

### 方法 1：使用配置文件（推荐）

在 `config.json` 中配置：

```json
{
  "benchmark": {
    "raw_file_dir": "data/app_data/lammps/16node-64proc-10iter-1000atom-default-20251219_140731",
    "csv_file": "data/processed/lammps/16node-64proc-1000atom-10interation.csv"
  }
}
```

运行：

```bash
cd /thfs3/home/xjtu_cx/myy/api/tune_ucx_with_config
python src/process/process.py
```

### 方法 2：指定配置文件

```bash
python src/process/process.py config/custom_config.json
```

### 方法 3：直接指定输入输出路径

```bash
python src/process/process.py \
    data/app_data/lammps/raw_data/ \
    data/processed/lammps/output.csv
```

### 方法 4：作为 Python 模块使用

```python
from src.process import process_from_config, process_raw_data

# 使用配置文件
df = process_from_config("config/config.json")

# 或直接指定路径
df = process_raw_data(
    raw_file_dir="data/app_data/lammps/raw_data",
    output_file="data/processed/lammps/output.csv"
)
```

## 📝 输出示例

```
======================================================================
Starting data processing pipeline
======================================================================
>>> Reading data from: data/app_data/lammps/16node-64proc-10iter-1000atom-default-20251219_140731
>>> Found 64 log files
    - Loaded: log-0.csv (1046 rows)
    - Loaded: log-1.csv (1046 rows)
    ...
>>> Total rows loaded: 66944
>>> Calculating total_size = sendsize * sendcount
>>> total_size range: [0.0, 27743.999999992462]
>>> Aggregating by comm_type and total_size
>>> Unique comm_types: 1
>>> Total unique (comm_type, total_size) pairs: 19
>>> Adding index within each comm_type
    - comm_type 51: 19 unique total_sizes
>>> Saving processed data to: data/processed/lammps/output.csv
    - Created directory: data/processed/lammps
>>> Successfully saved 19 rows
>>> Output columns: ['comm_type', 'index', 'total_size', 'count']
======================================================================
Data processing completed successfully!
======================================================================
```

## 🔧 处理逻辑详解

### 1. 读取所有日志文件

- 自动识别目录下所有 `log-*.csv` 文件
- 支持任意数量的日志文件
- 合并所有数据到单个 DataFrame

### 2. 计算 total_size

```python
total_size = sendsize × sendcount
```

表示每次通信的实际数据量。

### 3. 分组统计

按 `(comm_type, total_size)` 分组，统计每组出现的次数：

```
comm_type=51, total_size=1200.0 → 出现 1472 次 → count=1472
comm_type=51, total_size=1600.0 → 出现 1472 次 → count=1472
```

### 4. 添加索引

在每个 `comm_type` 内，按 `total_size` 升序排序，添加 `index`：

```
comm_type=51:
  index=1, total_size=0.0
  index=2, total_size=1200.0
  index=3, total_size=1600.0
  ...
```

## ⚠️ 注意事项

1. **依赖包**：需要安装 `pandas`
   ```bash
   pip install pandas
   ```

2. **文件命名**：原始数据文件必须匹配 `log-*.csv` 格式

3. **必需列**：原始数据必须包含以下列：
   - `comm_type`
   - `sendsize`
   - `sendcount`

4. **目录创建**：输出目录如果不存在会自动创建

## 🎯 集成到调优流程

### 完整工作流

```bash
# 1. 运行原始测试程序，生成 log-*.csv 文件
# 2. 处理数据
python src/process/process.py

# 3. 运行 OpenTuner 调优
python src/tune_ucx_latency_ping_with_avg_time.py
```

### 在代码中集成

可以在调优脚本中自动调用数据处理：

```python
from src.process import process_from_config

# 在调优前自动处理数据
print("Processing raw data...")
process_from_config("config/config.json")
print("Starting optimization...")
```

## 📈 性能说明

- 处理 64 个文件（约 66,000 行数据）：< 1 秒
- 内存占用：取决于数据量，一般 < 100MB
- 输出文件大小：通常 < 1KB（高度压缩的统计数据）

## 🐛 故障排除

### 错误：No log-*.csv files found

**原因**：指定目录下没有匹配的文件

**解决**：
- 检查 `raw_file_dir` 路径是否正确
- 确认文件名格式为 `log-0.csv`, `log-1.csv` 等

### 错误：KeyError: 'sendsize'

**原因**：CSV 文件缺少必需的列

**解决**：检查原始数据文件格式，确保包含所有必需列

### 错误：Config file not found

**原因**：配置文件路径错误

**解决**：使用正确的相对或绝对路径
