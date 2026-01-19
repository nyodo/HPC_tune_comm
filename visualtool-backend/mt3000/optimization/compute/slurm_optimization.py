from flask import Blueprint, jsonify, request
import os
import uuid

from api.mt3000_connect import Mt3000Client
from mt3000.config_loader import get_mt3000_client_config


slurm_optimization_bp = Blueprint("mt3000_optimization_compute_slurm", __name__)


_REMOTE_WORK_DIR_TMPL = "/thfs3/home/{username}/MCJM-Toolkits/src/COMP"


@slurm_optimization_bp.route("/mt3000/optimization/compute/slurm", methods=["POST"])
def submit_mt3000_optimization_compute_slurm():
    if "file" not in request.files:
        return jsonify({"error": "No file part (expected field name: file)"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    lower = file.filename.lower()
    if not (lower.endswith(".sh") or lower.endswith(".bash")):
        return jsonify({"error": "Unsupported file type (expected .sh/.bash)"}), 400

    base_dir = os.path.dirname(__file__)
    upload_dir = os.path.join(base_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    unique_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    save_name = f"{unique_id}{ext or '.sh'}"
    save_path = os.path.join(upload_dir, save_name)
    file.save(save_path)

    client_cfg = get_mt3000_client_config()
    username = client_cfg.get("username", "")

    remote_dir = _REMOTE_WORK_DIR_TMPL.format(username=username)
    remote_path = f"{remote_dir}/slurm.sh"

    try:
        client = Mt3000Client(
            hostname=client_cfg.get("hostname"),
            username=client_cfg.get("username"),
            password=client_cfg.get("password"),
            proxy_host=client_cfg.get("proxy_host", "127.0.0.1"),
            proxy_port=client_cfg.get("proxy_port", 1080),
            ssh_port=client_cfg.get("ssh_port", 22),
            connect_timeout=client_cfg.get("connect_timeout", 10),
        )
        try:
            client.run_command(f"mkdir -p {remote_dir}")
            client.upload_config(save_path, remote_path)
        finally:
            client.close()
    except Exception as e:
        return jsonify({"error": f"Upload to MT3000 failed: {e}", "localPath": save_path}), 500

    return jsonify({
        "message": "slurm.sh received and uploaded to MT3000",
        "filename": file.filename,
        "localPath": save_path,
        "remotePath": remote_path,
    }), 200
