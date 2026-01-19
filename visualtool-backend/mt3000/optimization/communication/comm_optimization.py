"""
COMM 通信调优模块的路由处理（Optimization/communication）
"""
import os
import json
import traceback
import time
from flask import Blueprint, request, jsonify, current_app

from api.mt3000_connect import Mt3000Client
from mt3000.config_loader import get_mt3000_client_config, get_remote_paths_config

from .utils import sanitize_filename, unique_path, normalize_comm_payload


comm_optimization_bp = Blueprint("comm_optimization", __name__)


@comm_optimization_bp.route("/comm-collection", methods=["POST"])
def handle_comm_collection():
    """提交通信调优任务"""
    client_config = get_mt3000_client_config()
    Mt3000Client1 = Mt3000Client(
        hostname=client_config.get("hostname"),
        username=client_config.get("username"),
        password=client_config.get("password"),
        proxy_host=client_config.get("proxy_host", "127.0.0.1"),
        proxy_port=client_config.get("proxy_port", 1080),
        ssh_port=client_config.get("ssh_port", 22),
        connect_timeout=client_config.get("connect_timeout", 10),
    )

    # 用于标记任务是否已提交
    task_submitted = False
    job_name = None
    log_file_path = None
    remote_result_dir = None
    
    try:
        form_data = request.get_json(silent=True)
        if not form_data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        normalized = normalize_comm_payload(form_data)

        # 必须提供 raw_dir_name + raw_files（多份 CSV）
        raw_dir_name = str(form_data.get("raw_dir_name") or "").strip()
        processed_subdir = str(form_data.get("processed_subdir") or "").strip()
        raw_files = form_data.get("raw_files") or []

        if not raw_dir_name:
            return jsonify({"status": "error", "message": "raw_dir_name is required"}), 400
        if not processed_subdir:
            return jsonify({"status": "error", "message": "processed_subdir is required"}), 400
        if not isinstance(raw_files, list) or len(raw_files) == 0:
            return jsonify({"status": "error", "message": "raw_files (list) is required"}), 400

        job_name = normalized.get("name") or f"comm_tune_{int(time.time())}"
        safe_job_name = sanitize_filename(job_name, default="comm_tune")
        filename = f"{safe_job_name}_config.json"

        upload_folder = current_app.config.get("UPLOAD_FOLDER", "upfile")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = unique_path(upload_folder, filename)
        server_filename = os.path.basename(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

        remote_paths = get_remote_paths_config("communication")
        work_dir = remote_paths.get(
            "work_dir",
            "/thfs3/home/{username}/MCJM-Toolkits/src/COMM/tune_ucx_with_opentuner",
        )
        work_dir = work_dir.format(username=Mt3000Client1.username)

        # 1) 上传“可追溯”配置文件到 config_dir（按时间戳保留）
        remote_dir = remote_paths.get(
            "config_dir",
            "/thfs3/home/{username}/MCJM-Toolkits/src/COMM/tune_ucx_with_opentuner/config/",
        )
        remote_dir = remote_dir.format(username=Mt3000Client1.username)
        Mt3000Client1.run_command(f"mkdir -p {remote_dir}")
        remote_json = os.path.join(remote_dir, server_filename)
        Mt3000Client1.upload_config(file_path, remote_json)

        # 2) 批量上传原始 CSV 文件到 data/app_data/<raw_dir_name>/（完全隔离在 tune_ucx_with_opentuner 下）
        uploaded_files_info = []
        remote_raw_base = remote_paths.get(
            "data_dir",
            "/thfs3/home/{username}/MCJM-Toolkits/src/COMM/tune_ucx_with_opentuner/data/app_data/",
        ).format(username=Mt3000Client1.username)
        remote_raw_dir_abs = os.path.join(remote_raw_base, raw_dir_name)
        Mt3000Client1.run_command(f"mkdir -p {remote_raw_dir_abs}")

        for item in raw_files:
            if not isinstance(item, dict):
                continue
            local_path = item.get("file_path")
            filename = item.get("filename") or (os.path.basename(local_path) if local_path else None)
            if not local_path or not filename:
                continue
            if not os.path.exists(local_path):
                continue

            remote_path_abs = os.path.join(remote_raw_dir_abs, os.path.basename(filename))
            Mt3000Client1.upload_config(local_path, remote_path_abs)
            uploaded_files_info.append({"type": "raw_csv", "remote_path": remote_path_abs})

        # 写入给调优程序用的相对路径（示例：data/app_data/astro_demo_16_node）
        normalized["benchmark"]["raw_file_dir"] = f"data/app_data/{raw_dir_name}"

        # 处理后 CSV 为调优临时文件：data/processed/<processed_subdir>/<raw_dir_name>.csv
        remote_processed_base = remote_paths.get(
            "processed_dir",
            "/thfs3/home/{username}/MCJM-Toolkits/src/COMM/tune_ucx_with_opentuner/data/processed/",
        ).format(username=Mt3000Client1.username)
        remote_processed_subdir_abs = os.path.join(remote_processed_base, processed_subdir)
        Mt3000Client1.run_command(f"mkdir -p {remote_processed_subdir_abs}")
        normalized["benchmark"]["csv_file"] = f"data/processed/{processed_subdir}/{raw_dir_name}.csv"

        # 3) 重新保存配置（含远程路径），并复制为工作目录下固定的 config/config.json
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

        remote_config_dir = os.path.join(work_dir, "config")
        Mt3000Client1.run_command(f"mkdir -p {remote_config_dir}")
        remote_config_file = os.path.join(remote_config_dir, "config.json")
        Mt3000Client1.upload_config(file_path, remote_config_file)

        # 4) 将旧的 OpenTuner 日志移到 backup 子目录，避免刚启动时读到历史文件
        script_path_cfg = normalized.get("script_shell", {}).get("script_path", "src/scripts/run_latency_2_intra-Blade.sh")
        script_name = os.path.basename(str(script_path_cfg))
        scenario_name = script_name.replace("run_latency_", "").replace(".sh", "")
        opentuner_log_dir_cfg = normalized.get("opentuner", {}).get("opentuner_log_dir", "tune_result_avg")
        internal_log_dir = os.path.join(work_dir, opentuner_log_dir_cfg, scenario_name)
        backup_dir = os.path.join(internal_log_dir, "backup")
        # 创建目录并将旧日志移动到 backup，避免覆盖；若无旧日志则静默
        Mt3000Client1.run_command(
            (
                f"mkdir -p {internal_log_dir} {backup_dir} && "
                f"for f in {internal_log_dir}/opentuner_log_*.txt; do "
                f"  [ -e \"$f\" ] || continue; "
                f"  base=$(basename \"$f\"); "
                f"  mv \"$f\" {backup_dir}/$(date +%s)_$base; "
                f"done"
            ),
            timeout=30,
            read_output=False,
        )

        # 5) 后台执行（日志轮询）
        log_timestamp = int(time.time())
        log_filename = f"opentuner_log_{log_timestamp}.txt"
        log_dir = os.path.join(
            work_dir, normalized.get("opentuner", {}).get("opentuner_log_dir", "tune_result_avg")
        )
        log_file_path = os.path.join(log_dir, log_filename)
        Mt3000Client1.run_command(f"mkdir -p {log_dir}")

        # 重要：通过 SSH 非交互执行时，默认不会加载 /etc/profile、~/.bashrc，导致 module/mpicc/salloc 等命令找不到（exit 127）。
        # 用 bash -lc 强制走登录 shell 环境，再用 nohup 后台运行并把输出重定向到日志文件。
        cmd = (
            f"cd {work_dir} && "
            f"nohup bash -lc 'python3 src/tune_ucx_latency_ping_with_avg_time.py' "
            f"> {log_file_path} 2>&1 &"
        )
        print(f"[COMM-OPT] 准备执行命令: {cmd}")
        # 后台启动命令不读取 stdout/stderr，避免 paramiko read 超时（任务已启动但通道读会卡住）
        out, err = Mt3000Client1.run_command(cmd, timeout=30, read_output=False)
        print(f"[COMM-OPT] 命令执行完成，stdout: {out[:200] if out else 'None'}, stderr: {err[:200] if err else 'None'}")
        
        # 标记任务已提交成功
        task_submitted = True
        time.sleep(2)

        result_dir = normalized.get("script_shell", {}).get("result_dir", "result/result_2_Intra-Blade")
        remote_result_dir = os.path.join(work_dir, result_dir)

        response_data = {
            "file_path": file_path,
            "job_name": job_name,
            "server_filename": server_filename,
            "remote_config_path": remote_json,
            "remote_cmd": cmd,
            "uploaded_files": uploaded_files_info,
            "result_dir": remote_result_dir,
            "log_file_path": log_file_path,
            "output": out[:1000] if out else "",
            "error": err[:1000] if err else "",
        }

        return jsonify(
            {
                "status": "success",
                "message": "Communication tuning task created and submitted successfully",
                "data": response_data,
            }
        ), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[COMM-OPT] 异常发生: {str(e)}")
        print(f"[COMM-OPT] 完整堆栈:\n{error_trace}")
        
        # 如果任务已经提交，即使后续出错也返回成功（因为任务确实已经在运行）
        if task_submitted:
            print("[COMM-OPT] 警告：任务已提交但后续处理出错，返回成功响应")
            return jsonify(
                {
                    "status": "success",
                    "message": "Task submitted successfully (some post-submission steps may have failed)",
                    "data": {
                        "job_name": job_name or "unknown",
                        "log_file_path": log_file_path or "",
                        "result_dir": remote_result_dir or "",
                        "warning": f"Post-submission error: {str(e)}",
                    },
                }
            ), 200
        
        # 任务提交前失败，返回错误
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Failed to process request: {str(e)}",
                    "trace": error_trace,
                }
            ),
            500,
        )

    finally:
        if Mt3000Client1 is not None:
            try:
                Mt3000Client1.close()
            except Exception:
                pass


@comm_optimization_bp.route("/comm-status", methods=["POST"])
def handle_comm_status():
    """查询任务状态（日志 + 进程）"""
    try:
        data = request.get_json() or {}
        job_name = data.get("job_name")

        # 外层 nohup 日志（可选，用于 fallback）
        log_file_path = data.get("log_file_path")

        if not job_name:
            return jsonify({"status": "error", "message": "job_name required"}), 400

        client_config = get_mt3000_client_config()
        Mt3000Client1 = Mt3000Client(
            hostname=client_config.get("hostname"),
            username=client_config.get("username"),
            password=client_config.get("password"),
            proxy_host=client_config.get("proxy_host", "127.0.0.1"),
            proxy_port=client_config.get("proxy_port", 1080),
            ssh_port=client_config.get("ssh_port", 22),
            connect_timeout=client_config.get("connect_timeout", 10),
        )

        try:
            # 优先读取 Python 调优脚本自己的日志：
            # <work_dir>/<opentuner_log_dir>/<scenario>/opentuner_log_*.txt
            # scenario 来自 script_shell.script_path：run_latency_2_intra-Blade.sh -> 2_intra-Blade
            remote_paths = get_remote_paths_config("communication")
            work_dir = remote_paths.get(
                "work_dir",
                "/thfs3/home/{username}/MCJM-Toolkits/src/COMM/tune_ucx_with_opentuner",
            ).format(username=Mt3000Client1.username)

            config_path = os.path.join(work_dir, "config", "config.json")
            config_json_str, _ = Mt3000Client1.run_command(
                f"cat {config_path} 2>/dev/null || echo ''",
                timeout=10,
            )

            script_path = None
            opentuner_log_dir = None
            scenario_name = None
            if config_json_str.strip():
                try:
                    cfg = json.loads(config_json_str)
                    script_path = (cfg.get("script_shell", {}) or {}).get("script_path")
                    opentuner_log_dir = (cfg.get("opentuner", {}) or {}).get("opentuner_log_dir", "tune_result_avg")
                except Exception:
                    script_path = None
                    opentuner_log_dir = None

            if script_path:
                script_name = os.path.basename(str(script_path))
                scenario_name = script_name.replace("run_latency_", "").replace(".sh", "")

            chosen_log = ""
            log_output = ""
            if scenario_name and opentuner_log_dir:
                internal_log_dir = os.path.join(work_dir, str(opentuner_log_dir), str(scenario_name))
                latest_cmd = f"ls -t {internal_log_dir}/opentuner_log_*.txt 2>/dev/null | head -n 1"
                latest_file, _ = Mt3000Client1.run_command(latest_cmd, timeout=10)
                latest_file = (latest_file or "").strip()
                if latest_file:
                    chosen_log = latest_file
                    log_output, _ = Mt3000Client1.run_command(
                        f"tail -n 200 {latest_file} 2>/dev/null || echo 'Log file not found'",
                        timeout=10,
                    )

            # fallback：如果找不到内部日志，则读外层 nohup 日志
            if not log_output:
                if log_file_path:
                    chosen_log = log_file_path
                    tail_cmd = f"tail -n 200 {log_file_path} 2>/dev/null || echo 'Log file not found'"
                    log_output, _ = Mt3000Client1.run_command(tail_cmd, timeout=10)
                else:
                    log_output = ""

            check_process_cmd = (
                "ps aux | grep 'tune_ucx_latency_ping_with_avg_time.py' | grep -v grep || echo 'Process not running'"
            )
            process_status, _ = Mt3000Client1.run_command(check_process_cmd)

            return (
                jsonify(
                    {
                        "status": "success",
                        "log_output": log_output,
                        "log_source": chosen_log,
                        "process_status": process_status,
                        "is_running": "Process not running" not in process_status,
                    }
                ),
                200,
            )
        finally:
            Mt3000Client1.close()

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@comm_optimization_bp.route("/comm-results", methods=["POST"])
def handle_comm_results():
    """获取调优结果文件（目录列表 + best params）"""
    try:
        data = request.get_json() or {}
        result_dir = data.get("result_dir")

        if not result_dir:
            return jsonify({"status": "error", "message": "result_dir required"}), 400

        client_config = get_mt3000_client_config()
        Mt3000Client1 = Mt3000Client(
            hostname=client_config.get("hostname"),
            username=client_config.get("username"),
            password=client_config.get("password"),
            proxy_host=client_config.get("proxy_host", "127.0.0.1"),
            proxy_port=client_config.get("proxy_port", 1080),
            ssh_port=client_config.get("ssh_port", 22),
            connect_timeout=client_config.get("connect_timeout", 10),
        )

        try:
            list_cmd = f"ls -lh {result_dir} 2>/dev/null || echo 'Directory not found'"
            file_list, _ = Mt3000Client1.run_command(list_cmd)

            best_params_path = os.path.join(os.path.dirname(result_dir), "outputs", "ucx_best_params.json")
            read_params_cmd = f"cat {best_params_path} 2>/dev/null || echo 'Best params file not found'"
            best_params, _ = Mt3000Client1.run_command(read_params_cmd)

            return (
                jsonify(
                    {
                        "status": "success",
                        "file_list": file_list,
                        "best_params": best_params if "not found" not in best_params else None,
                    }
                ),
                200,
            )
        finally:
            Mt3000Client1.close()

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

