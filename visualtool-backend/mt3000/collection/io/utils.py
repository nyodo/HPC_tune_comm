"""
IO 数据采集相关的工具函数
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


_SIZE_RE = re.compile(r"^(-?\d+)\s*([kKmMgGtT])?$")


def _parse_value_token(tok: str) -> str:
    """解析值令牌，支持整数和大小单位（K/M/G/T）"""
    s = str(tok).strip()
    if not s:
        return ""

    # 纯整数
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))

    m = _SIZE_RE.match(s)
    if m and m.group(2):
        base = int(m.group(1))
        unit = m.group(2).lower()
        mul = 1
        if unit == "k":
            mul = 1024
        elif unit == "m":
            mul = 1024 ** 2
        elif unit == "g":
            mul = 1024 ** 3
        elif unit == "t":
            mul = 1024 ** 4
        return str(base * mul)

    # 兜底：保留原样字符串
    return s


def _split_tokens(text: str):
    """分割文本为令牌列表"""
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    parts = re.split(r"[,;\n\s]+", s)
    return [p.strip() for p in parts if p and p.strip()]


def _ensure_str_list(val):
    """确保值为字符串列表格式"""
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for x in val:
            px = _parse_value_token(x)
            if px != "":
                out.append(px)
        return out
    if isinstance(val, (int, float)):
        return [str(int(val))]
    if isinstance(val, str):
        toks = _split_tokens(val)
        return [_parse_value_token(t) for t in toks if _parse_value_token(t) != ""]
    # 其它类型：转字符串尝试
    toks = _split_tokens(str(val))
    return [_parse_value_token(t) for t in toks if _parse_value_token(t) != ""]


def _to_int_or_none(v):
    """转换为整数或 None"""
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


def normalize_payload_to_input_file_n(form_data: dict) -> dict:
    """将表单数据标准化为后端需要的 JSON 格式"""
    if not isinstance(form_data, dict):
        return {}

    mpi_mode = (form_data.get("MPI_mode") or "").strip().lower()

    # 基础字段：按你示例保留
    out = {
        "name": form_data.get("name"),
        "result_folder": form_data.get("result_folder"),
        "nodes": _to_int_or_none(form_data.get("nodes")),
        "MPI_mode": mpi_mode,
        "tasks_per_node": _to_int_or_none(form_data.get("tasks_per_node")),
        "run_command": form_data.get("run_command"),
        "model_name": form_data.get("model_name"),

        "iter_start": _to_int_or_none(form_data.get("iter_start")),
        "iter_end": _to_int_or_none(form_data.get("iter_end")) if form_data.get("iter_end") not in (None, "") else None,
        "train_mode": form_data.get("train_mode"),
        "save_rounds": _to_int_or_none(form_data.get("save_rounds")),
        "partition": form_data.get("partition"),

        "self_module": form_data.get("self_module", None),
        "self_export": form_data.get("self_export", None),
        "module_load": bool(form_data.get("module_load", True)),
    }

    # tasks_per_node 上限 16：后端兜底
    if out["tasks_per_node"] is not None:
        out["tasks_per_node"] = max(1, min(16, out["tasks_per_node"]))

    # -------- slurm_shell --------
    slurm_shell = form_data.get("slurm_shell") or {}
    if not isinstance(slurm_shell, dict):
        slurm_shell = {}

    out["slurm_shell"] = {
        "file_path": (slurm_shell.get("file_path") or ""),
        "parameters": {}
    }

    raw_params = slurm_shell.get("parameters") or {}
    if not isinstance(raw_params, dict):
        raw_params = {}

    if mpi_mode == "openmpi":
        # 仅 openmpi 保留/解析 parameters（全部转成 list[str]）
        params_out = {}

        # OMPI 常用项（可选：你如果想强制必填，在这里校验）
        for k in [
            "__OMPI_MCA_io_ompio_grouping_option",
            "__OMPI_MCA_io_ompio_num_aggregators",
            "__OMPI_MCA_io_ompio_cycle_buffer_size",
            "__OMPI_MCA_io_ompio_bytes_per_agg",
            "blocksize, transfersize, segment",
            "blocksize_transfersize_segment",  # 兼容你前端字段
        ]:
            if k in raw_params and raw_params.get(k) not in (None, ""):
                params_out[k] = _ensure_str_list(raw_params.get(k))

        # 如果前端用 blocksize_transfersize_segment 传，统一写成你示例里的 key
        if "blocksize_transfersize_segment" in params_out and "blocksize, transfersize, segment" not in params_out:
            params_out["blocksize, transfersize, segment"] = params_out.pop("blocksize_transfersize_segment")

        out["slurm_shell"]["parameters"] = params_out
    else:
        # mpich / posixmpi：强制 {}
        out["slurm_shell"]["parameters"] = {}

    input_keys = sorted([k for k in form_data.keys() if re.fullmatch(r"input_file_\d+", k)])
    if input_keys:
        for k in input_keys:
            block = form_data.get(k) or {}
            if not isinstance(block, dict):
                continue
            fp = (block.get("file_path") or "").strip()
            params = block.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params_out = {pk: _ensure_str_list(pv) for pk, pv in params.items() if pk and pv is not None}
            out[k] = {"file_path": fp, "parameters": params_out}
        return out

    arr = form_data.get("input_files") or []
    if isinstance(arr, list) and arr:
        idx = 1
        for item in arr:
            if not isinstance(item, dict):
                continue
            fp = (item.get("file_path") or "").strip()
            params = item.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}

            # 兼容你旧的 "__region, __runSteps, __dump_times"
            params_out = {}
            for pk, pv in params.items():
                if not pk:
                    continue
                params_out[pk] = _ensure_str_list(pv)

            out[f"input_file_{idx}"] = {"file_path": fp, "parameters": params_out}
            idx += 1
        return out

    # 3) 如果没有任何输入文件：仍然返回 out（后端可在这儿强制报错）
    return out
