# Config.json 配置文件说明

## 文件位置
`config/config.json`

## 配置结构

```json
{
  "cluster": {
    "partition": "thcp3",
    "nodes": ["cn22257", "cn22258"]
  },
  "benchmark": {
    "csv_file": "data/processed/misa/16node-64proc-step1-20251203_162824_statistics.csv",
    "comm_types": [55]
  }
}
```

## 配置项详解

### 1. cluster（集群配置）

#### partition
- **类型**: 字符串
- **说明**: SLURM 分区名称
- **用途**: 用于 `salloc` 命令申请节点时指定分区
- **示例**: `"thcp3"`

#### nodes
- **类型**: 字符串数组
- **说明**: 要使用的节点名称列表
- **要求**: 至少需要 2 个节点（用于跨节点通信测试）
- **用途**: 传递给 `salloc --nodelist` 参数
- **示例**: `["cn22257", "cn22258"]`

---

### 2. benchmark（基准测试配置）

#### csv_file
- **类型**: 字符串
- **说明**: MPI 通信模式数据的 CSV 文件路径（相对于项目根目录）
- **格式**: CSV 文件应包含以下列：
  - `comm_type`: 通信类型编号
  - `index`: 序列号
  - `total_size`: 消息大小（字节）
  - `count`: 该大小消息的调用次数
- **默认值**: `"data/processed/misa/16node-64proc-step1-20251203_162824_statistics.csv"`
- **示例**: 
  ```
  "csv_file": "data/processed/lammps/16node-64proc-1000atom-10interation.csv"
  ```

#### comm_types
- **类型**: 整数数组
- **说明**: 要处理的通信类型列表（过滤条件）
- **用途**: 只处理 CSV 文件中 `comm_type` 列匹配这些值的记录
- **默认值**: `[55]`
- **示例**: 
  ```json
  "comm_types": [55, 56, 57]  # 处理多种通信类型
  ```

---

## 使用示例

### 示例 1: 默认配置
```json
{
  "cluster": {
    "partition": "thcp3",
    "nodes": ["cn22257", "cn22258"]
  },
  "benchmark": {
    "csv_file": "data/processed/misa/16node-64proc-step1-20251203_162824_statistics.csv",
    "comm_types": [55]
  }
}
```

### 示例 2: 使用不同的数据集
```json
{
  "cluster": {
    "partition": "gpu_partition",
    "nodes": ["gpu01", "gpu02", "gpu03"]
  },
  "benchmark": {
    "csv_file": "data/processed/lammps/custom_benchmark.csv",
    "comm_types": [55, 56]
  }
}
```

### 示例 3: 测试多种通信类型
```json
{
  "cluster": {
    "partition": "thcp3",
    "nodes": ["cn22257", "cn22258"]
  },
  "benchmark": {
    "csv_file": "data/processed/misa/16node-64proc-step1-20251203_162824_statistics.csv",
    "comm_types": [55, 56, 57, 58]
  }
}
```

---

## 工作原理

1. **Shell 脚本读取配置**
   - `run_latency_2_intra-Blade.sh` 启动时读取 `config.json`
   - 使用 Python 解析 JSON 文件
   - 将配置值设置为环境变量：
     - `UCX_BENCHMARK_CSV_FILE`: CSV 文件的完整路径
     - `UCX_BENCHMARK_COMM_TYPES`: 逗号分隔的通信类型列表

2. **C 程序读取环境变量**
   - `ucx_latency_ping.c` 在启动时调用 `load_config()` 函数
   - 从环境变量 `UCX_BENCHMARK_CSV_FILE` 读取 CSV 文件路径
   - 从环境变量 `UCX_BENCHMARK_COMM_TYPES` 读取通信类型列表
   - 如果环境变量未设置，则使用默认值

3. **灵活性**
   - 无需重新编译 C 程序即可更换数据集
   - 无需修改脚本即可调整节点配置
   - 所有配置集中在一个文件中，便于管理

---

## 注意事项

1. **路径约定**
   - `csv_file` 使用相对于项目根目录的相对路径
   - Shell 脚本会自动添加 `${ROOT_DIR}/` 前缀

2. **文件存在性检查**
   - Shell 脚本会检查 `config.json` 是否存在
   - C 程序会检查 CSV 文件是否存在并给出明确的错误信息

3. **数组格式**
   - `nodes` 和 `comm_types` 必须使用 JSON 数组格式 `[...]`
   - `comm_types` 中的值必须是整数

4. **向后兼容**
   - 如果 `config.json` 中缺少 `benchmark` 部分，程序会使用默认值
   - 确保与旧版本配置文件的兼容性

---

## 验证配置

可以使用以下 Python 命令验证配置文件格式：

```bash
python -c "import json; print(json.dumps(json.load(open('config/config.json')), indent=2))"
```

预期输出应该是格式化后的 JSON，没有错误信息。

---

## 常见问题

### Q: CSV 文件路径错误怎么办？
A: 检查以下几点：
- 路径是否相对于项目根目录
- 文件是否真实存在
- 文件名拼写是否正确

### Q: 如何使用绝对路径？
A: 可以直接在配置中使用绝对路径：
```json
"csv_file": "/absolute/path/to/your/data.csv"
```

### Q: comm_types 可以为空吗？
A: 不建议为空。如果为空数组 `[]`，程序会使用默认值 `[55]`。

### Q: 如何添加新的配置项？
A: 可以在 `benchmark` 下添加新字段，但需要同时修改 Shell 脚本和 C 程序代码以支持新配置项。
