"""
COMM 通信调优相关的工具函数（Optimization 模块下）
"""
import os
import re


def sanitize_filename(name: str, default: str = "unknown") -> str:
    """清理文件名，移除非法字符"""
    if name is None:
        name = ""
    name = str(name).strip()
    if not name:
        name = default

    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", "_", name)

    if name in {".", ".."}:
        name = default

    return name[:100]


def unique_path(folder: str, filename: str) -> str:
    """生成唯一文件路径，避免重名覆盖"""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{base}_{i}{ext}")
        i += 1
    return candidate


def _to_int_or_none(v):
    """转换为整数或 None"""
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


def _to_int_list(v):
    """转换为整数列表"""
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [int(x) for x in v if isinstance(x, (int, str)) and str(x).strip().isdigit()]
    if isinstance(v, str):
        # 支持逗号、分号、空格分隔
        parts = re.split(r"[,;\\s]+", v.strip())
        result = []
        for p in parts:
            p = p.strip()
            if p.isdigit():
                result.append(int(p))
        return result
    return []


def normalize_comm_payload(form_data: dict) -> dict:
    """将前端表单数据标准化为后端需要的 config.json 格式"""
    if not isinstance(form_data, dict):
        return {}

    config = {}

    # 1. cluster 配置
    cluster = {}
    cluster["partition"] = str(form_data.get("partition") or "thcp3")

    nodes = form_data.get("nodes")
    if nodes:
        if isinstance(nodes, list):
            cluster["nodes"] = [str(n) for n in nodes]
        elif isinstance(nodes, str):
            cluster["nodes"] = [n.strip() for n in nodes.split(",") if n.strip()]
        else:
            cluster["nodes"] = []
    else:
        cluster["nodes"] = ["cn27584", "cn27585"]

    config["cluster"] = cluster

    # 2. benchmark 配置
    benchmark = {}

    # 前端只上传 raw_files（多份 CSV），并指定 raw_dir_name / processed_subdir
    raw_dir_name = str(form_data.get("raw_dir_name") or "").strip()
    processed_subdir = str(form_data.get("processed_subdir") or "").strip()

    # 写入给调优程序使用的相对路径（相对 work_dir）
    benchmark["raw_file_dir"] = f"data/app_data/{raw_dir_name}" if raw_dir_name else "data/app_data/"
    if processed_subdir and raw_dir_name:
        benchmark["csv_file"] = f"data/processed/{processed_subdir}/{raw_dir_name}.csv"
    else:
        benchmark["csv_file"] = "data/processed/"

    comm_types = _to_int_list(form_data.get("comm_types"))
    if not comm_types:
        comm_types = [55]
    benchmark["comm_types"] = comm_types

    config["benchmark"] = benchmark

    # 3. script_shell 配置
    script_shell = {
        "script_path": form_data.get("script_path", "src/scripts/run_latency_2_intra-Blade.sh"),
        "result_dir": form_data.get("result_dir", "result/result_2_Intra-Blade"),
        "save_shell_output": bool(form_data.get("save_shell_output", False)),
    }
    config["script_shell"] = script_shell

    # 4. opentuner 配置
    opentuner = {
        "test_limit": _to_int_or_none(form_data.get("test_limit")) or 50,
        "no_dups": bool(form_data.get("no_dups", True)),
        "opentuner_log_dir": form_data.get("opentuner_log_dir", "tune_result_avg"),
        "save_opentuner_log": bool(form_data.get("save_opentuner_log", True)),
    }
    config["opentuner"] = opentuner

    # 5. 作业名称
    if form_data.get("name"):
        config["name"] = form_data.get("name")

    return config

