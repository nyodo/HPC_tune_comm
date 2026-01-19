"""
IO 数据采集模块的路由处理
"""
import os
import json
import traceback
from flask import Blueprint, request, jsonify, current_app
from api.mt3000_connect import Mt3000Client
from mt3000.config_loader import (
    get_mt3000_client_config,
    get_remote_paths_config
)
from .utils import (
    sanitize_filename,
    unique_path,
    normalize_payload_to_input_file_n
)

# 创建蓝图
io_collection_bp = Blueprint('io_collection', __name__)


@io_collection_bp.route('/io-collection', methods=['POST'])
def handle_io_collection():
    """处理 IO 数据采集请求"""
    # 从配置文件加载 MT3000 客户端配置
    client_config = get_mt3000_client_config()
    Mt3000Client1 = Mt3000Client(
        hostname=client_config.get('hostname'),
        username=client_config.get('username'),
        password=client_config.get('password'),
        proxy_host=client_config.get('proxy_host', '127.0.0.1'),
        proxy_port=client_config.get('proxy_port', 1080),
        ssh_port=client_config.get('ssh_port', 22),
        connect_timeout=client_config.get('connect_timeout', 10),
    )

    try:
        # 1. 解析前端 JSON 表单
        form_data = request.get_json(silent=True)
        if not form_data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        # 允许前端指定执行步骤，默认 1 = 数据采集
        step = str(form_data.get('step', '1'))
        if step not in {"1", "2", "3", "4"}:
            return jsonify({'status': 'error', 'message': 'Invalid step, must be 1~4'}), 400

        # 2. 标准化为后端需要的 JSON 格式
        normalized = normalize_payload_to_input_file_n(form_data)

        job_name = normalized.get('name') or 'unknown'
        safe_job_name = sanitize_filename(job_name, default="unknown")
        filename = f"{safe_job_name}.json"

        # 3. 写到本地临时目录
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'upfile')
        os.makedirs(upload_folder, exist_ok=True)

        file_path = unique_path(upload_folder, filename)  # 避免重名覆盖
        server_filename = os.path.basename(file_path)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

        print(f"[IOJM] IO Collection config saved to local: {file_path}")

        # 4. 连接 MT3000 平台并上传 JSON 配置
        # 从配置文件加载远程路径配置（IO 模块）
        remote_paths = get_remote_paths_config('io')
        remote_dir = remote_paths.get('config_dir', '/thfs3/home/{username}/MCJM-Toolkits/config/IOJM/')
        remote_dir = remote_dir.format(username=Mt3000Client1.username)
        remote_json = os.path.join(remote_dir, server_filename)
        print(f"[IOJM] 开始上传 {file_path} 到远程 {remote_json} ...")
        Mt3000Client1.upload_config(file_path, remote_json)
        print("[IOJM] 文件上传完成！")

        # 5. 在远程调用 IOJM main.py，执行对应步骤
        # -u 传配置文件路径
        # -s 传步骤：1=采集数据，2=数据处理，3=建模，4=预测
        remote_main = remote_paths.get('main_script', '/thfs3/home/{username}/MCJM-Toolkits/src/IOJM/main.py')
        remote_main = remote_main.format(username=Mt3000Client1.username)
        work_dir = remote_paths.get('work_dir', '/thfs3/home/{username}/MCJM-Toolkits/src/IOJM')
        work_dir = work_dir.format(username=Mt3000Client1.username)
        cmd = (
            f"cd {work_dir} && "
            f"python {remote_main} -u '{remote_json}' -s {step}"
        )
        print(f"[IOJM] 在远程执行命令: {cmd}")
        out, err = Mt3000Client1.run_command(cmd)
        if out:
            print("[IOJM] 远程任务输出:\n", out)
        if err:
            print("[IOJM] 远程任务错误输出:\n", err)

        # 6. （可选）写入数据库
        file_id = None
        # try:
        #     file_id = add_file({
        #         'filename': filename,
        #         'server_filename': server_filename,
        #     })
        # except Exception as db_error:
        #     print(f"[IOJM] Database save warning: {db_error}")
        #     file_id = None

        return jsonify({
            'status': 'success',
            'message': 'IO collection task created and submitted successfully',
            'data': {
                'file_id': file_id,
                'file_path': file_path,
                'job_name': job_name,
                'server_filename': server_filename,
                'remote_path': remote_json,
                'step': step,
                'remote_cmd': cmd,
            }
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Failed to process request: {str(e)}',
            'trace': traceback.format_exc()
        }), 500

    finally:
        # 确保 SSH 连接被关闭
        if Mt3000Client1 is not None:
            try:
                Mt3000Client1.close()
                print("[IOJM] SSH 连接已关闭")
            except Exception as close_err:
                print("[IOJM] 关闭 SSH 连接时异常:", close_err)
