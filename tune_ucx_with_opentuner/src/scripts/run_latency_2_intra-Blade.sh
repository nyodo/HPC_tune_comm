#!/bin/bash
# =================================================================
# 环境加载部分 (根据你的新环境修改)
# =================================================================
module purge
module load compiler/rocm/dtk/25.04
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.4.1/gcc-7.3.1
module load compiler/c.24.1

# 检查编译器和MPI是否加载成功
which mpicc > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo ">>> Error: mpicc not found. Check module loading."
    exit 1
fi

# =================================================================
# OpenTuner Runner Script (Corrected for Interactive Slurm)
# =================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MPI_PROGRAM="${ROOT_DIR}/src/ping/ucx_latency_ping.c"
EXECUTABLE="${ROOT_DIR}/src/ping/ucx_latency_ping_2"
RESULT_ROOT="${ROOT_DIR}/result"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S_%N")

# === 编译 ===
# 注意：HPC-X (OpenMPI) 也是使用 mpicc。
# -lm 是链接数学库，通常都需要。
echo ">>> Compiling with $(which mpicc)..."
mpicc -o ${EXECUTABLE} ${MPI_PROGRAM} -lm
if [ $? -ne 0 ]; then
    echo ">>> Error: Compilation failed!"
    exit 1
fi

SCENARIO_NAME="2_Intra-Blade"
OUTPUT_DIR="${RESULT_ROOT}/result_${SCENARIO_NAME}"
OUTPUT_FILE="${OUTPUT_DIR}/results_${SCENARIO_NAME}_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}"

echo ">>> Scenario: ${SCENARIO_NAME}"
echo ">>> Output: ${OUTPUT_FILE}"

# === 从 config.json 读取 benchmark 配置 ===
CONFIG_JSON="${ROOT_DIR}/config/config.json"

if [ ! -f "${CONFIG_JSON}" ]; then
    echo ">>> Error: Missing ${CONFIG_JSON}"
    exit 1
fi

# 检查 python 是否可用 (CentOS 7/Devtoolset 环境下 python 可能是 2.7，python3 可能是 3.x)
# 这里尝试优先使用 python3，如果没有则使用 python
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

# 读取 CSV 文件路径和 comm_types
CSV_FILE=$($PYTHON_CMD -c "import json; config=json.load(open('${CONFIG_JSON}')); print(config.get('benchmark', {}).get('csv_file', 'data/processed/misa/16node-64proc-step1-20251203_162824_statistics.csv'))")
COMM_TYPES=$($PYTHON_CMD -c "import json; config=json.load(open('${CONFIG_JSON}')); print(','.join(map(str, config.get('benchmark', {}).get('comm_types', [55]))))")

# 设置环境变量供 MPI 程序读取
export UCX_BENCHMARK_CSV_FILE="${ROOT_DIR}/${CSV_FILE}"
export UCX_BENCHMARK_COMM_TYPES="${COMM_TYPES}"

echo ">>> Using CSV file: ${UCX_BENCHMARK_CSV_FILE}"
echo ">>> Filtering COMM_TYPEs: ${COMM_TYPES}"

# === 自动申请节点并执行 ===
if [ -z "${SLURM_JOB_ID}" ]; then
    # 不在 SLURM 作业环境中，需要申请节点
    echo ">>> Not in SLURM environment, requesting nodes..."

    PARTITION=$($PYTHON_CMD -c "import json;print(json.load(open('${CONFIG_JSON}'))['cluster']['partition'])")
    NODELIST=$($PYTHON_CMD -c "import json;print(','.join(json.load(open('${CONFIG_JSON}'))['cluster']['nodes']))")

    echo ">>> Allocating nodes: ${NODELIST} on partition ${PARTITION}"

    # 使用 salloc 申请节点，然后在分配的环境中重新执行此脚本
    # 注意：HPC-X 环境下 salloc 行为通常一致
    salloc -p "${PARTITION}" -N 2 --nodelist="${NODELIST}" bash "${BASH_SOURCE[0]}"

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ">>> Success! Check output at: ${OUTPUT_DIR}"
        ls -lh "${OUTPUT_DIR}"
    else
        echo ">>> Error: salloc execution failed with code ${EXIT_CODE}"
    fi
    exit ${EXIT_CODE}
else
    # 已经在 SLURM 作业环境中，执行实际任务
    echo ">>> Running in SLURM job ${SLURM_JOB_ID}"
    echo ">>> Allocated nodes: ${SLURM_JOB_NODELIST}"

    cd "${ROOT_DIR}"

    # === 执行命令调整 ===
    # HPC-X 基于 OpenMPI。
    # 如果 srun 遇到问题（如 MPI 初始化失败），可能需要添加 --mpi=pmi2 或 --mpi=pmix
    # 这里保持默认，通常 Slurm 原生支持 OpenMPI
    echo ">>> Executing: srun -n 4 --ntasks-per-node=2 ${EXECUTABLE}"
    
    # 某些 HPC-X 版本可能需要加载 UCX 相关的库路径，module load 通常已处理好。
    # 如果遇到 UCX 警告，可能需要设置 UCX_TLS=all 或类似的变量，视具体硬件而定。

    # 根据环境变量决定是否保存日志
    if [ -n "${UCX_DISABLE_SHELL_LOG}" ] && [ "${UCX_DISABLE_SHELL_LOG}" = "1" ]; then
        # 不保存日志，但需要临时文件用于解析时间
        echo ">>> Shell output logging: DISABLED"
        TEMP_FILE="${OUTPUT_DIR}/.temp_result_${TIMESTAMP}.log"
        srun -n 4 --ntasks-per-node=2 --kill-on-bad-exit=1 "${EXECUTABLE}" > "${TEMP_FILE}" 2>&1

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -eq 0 ]; then
            mv "${TEMP_FILE}" "${OUTPUT_FILE}"
            echo ">>> Execution completed successfully"
        else
            rm -f "${TEMP_FILE}"
            echo ">>> Error: Execution failed with code ${EXIT_CODE}"
        fi
    else
        # 正常保存所有日志文件
        echo ">>> Shell output logging: ENABLED"
        srun -n 4 --ntasks-per-node=2 --kill-on-bad-exit=1 "${EXECUTABLE}" > "${OUTPUT_FILE}" 2>&1

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -eq 0 ]; then
            echo ">>> Execution completed successfully"
            echo ">>> Output saved to: ${OUTPUT_FILE}"
        else
            echo ">>> Error: Execution failed with code ${EXIT_CODE}"
        fi
    fi

    exit ${EXIT_CODE}
fi