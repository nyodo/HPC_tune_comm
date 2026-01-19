# 实例化对象
from flask import Flask
from flask_migrate import Migrate as migrate

from settings import SECRET_KEY
from flask_cors import CORS
from datetime import timedelta
import jwt
import os
import uuid
import time
import base64
import json

# 导入数据库操作的子路由的蓝图
from api.user import user_bp
from api.dcu import *
from api.file import *
from models import *

# 导入自己实现的接口，hip_code_modeling是访存组工具。llm为大模型调用。analyze_hip是建模。等后续建模组统一，再替换成新的。
# from hip_code_modeling import llm
# from hip_code_modeling.deploy import analyze_hip

app = Flask(__name__, static_folder='static')

CORS(app, resources={r"/*": {"origins": "*"}})

model_path = '../../github/LLM/Atom-7B-Chat'
gpu_id = '0'


class Config(object):
    """配置参数"""
    # sqlalchemy的配置参数
    SQLALCHEMY_DATABASE_URI = "mysql://root:123456@127.0.0.1:3306/visualtool"
    # 设置每次请求结束后会自动提交数据库中的改动，一般都设置手动保存
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    # 设置sqlalchemy自动更新跟踪数据库
    SQLALCHEMY_TRACK_MODIFICATIONS = True


# 连接数据库
app.config.from_object(Config)

db.init_app(app)
grate = migrate(app, db)

# 是子路由
app.register_blueprint(user_bp, url_prefix='/users')
app.register_blueprint(dcu_bp, url_prefix='/dcus')
app.register_blueprint(file_bp, url_prefix='/files')


def add_admin():
    # 查询是否有名为 'admin' 的管理员
    existing_admin = User.query.filter_by(username='admin', is_admin=True).first()
    if existing_admin is None:
        # 如果不存在，则插入新的管理员记录
        admin = User(username='admin', password='admin', is_admin=True)
        db.session.add(admin)
        db.session.commit()
        print("Admin added successfully.")
    else:
        print("Admin already exists.")


# 验证token
def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return "Token has expired"
    except jwt.InvalidTokenError:
        return "Invalid token"


# 前台返回格式
ret = {
    "data": {},
    "meta": {
        "status": 200,
        "message": "注册成功"
    }
}


@app.route('/')
def index():
    # 管理员表添加一条记录，用于前端登录 用户名admin 密码admin
    add_admin()
    source_code= "__global__ void mm2_kernel1(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *A, DATA_TYPE *B)\n{\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tint i = blockIdx.y * blockDim.y + threadIdx.y;\n\n\tif ((i < _PB_NI) && (j < _PB_NJ))\n\t{ \n\t\ttmp[i * NJ + j] = 0;\n\t\tint k;\n\t\tfor (k = 0; k < _PB_NK; k++)\n\t\t{\n\t\t\ttmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];\n\t\t}\n\t}\n}"
    start_time = time.time()  # 记录开始时间
    # analysis = analyze_hip(source_code)
    analysis = "分析结果"
    end_time = time.time()    # 记录结束时间
    execution_time = end_time - start_time
    print(f"建模花费时间: {execution_time} 秒")
    start_time = time.time()  # 记录开始时间
    # improve=llm.improve_hip_code(source_code)
    improve = "调优后结果"
    end_time = time.time()    # 记录结束时间
    execution_time = end_time - start_time
    print(f"调优花费时间: {execution_time} 秒")
    return analysis
    # return 'Hello!'


# 测试接口
@app.route('/test', methods=['GET'])
def handle_test():
    response = {
        'status': 'success',
        'message': 'Form submitted successfully!',
        'data': 'hello'
    }
    return jsonify(response), 200


# 处理意见反馈
@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.json  # 获取前端发送的 JSON 数据
    name = data.get('name')
    email = data.get('email')
    subject = data.get('subject')
    message = data.get('message')

    # 在这里处理你的表单数据，例如保存到数据库或发送电子邮件
    print(f"Received form submission: {data}")

    # 模拟一个简单的成功响应
    response = {
        'status': 'success',
        'message': 'Form submitted successfully!',
        'data': data
    }

    return jsonify(response), 200

# 新增函数：判断 type 是否为 file，并返回文件路径
def get_file_path(item):
    if item and item.get('type') == 'file':
        file_info = item.get('file')
        if file_info and 'filePath' in file_info:
            return file_info['filePath']
    return None

def get_content(item):
    if not item or 'type' not in item:
        return None
    if item['type'] == 'text':
        return item.get('text')
    elif item['type'] == 'file':
        file_info = item.get('file')
        if file_info and 'filePath' in file_info:
            file_path = file_info['filePath']
            full_path = os.path.join(os.getcwd(), file_path)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                return ''.join(lines)
            except Exception as e:
                print(f"Error reading file: {e}")
                return None
    return None

# 分析DCU平台代码
@app.route('/dcu_code', methods=['POST'])
def handle_dcu():
    data = request.json
    user_id = data['user_id']
    
    # 获取 code, ir, cfg 的内容
    code_content = get_content(data['code'])
    # ir_content = get_content(data['ir'])
    # cfg_content = get_content(data['cfg'])
    
    # 获取 code, ir, cfg, dynamicData 的文件路径
    code_path = get_file_path(data.get('code'))
    # ir_path = get_file_path(data.get('ir'))
    # cfg_path = get_file_path(data.get('cfg'))
    # dynamic_data_path = get_file_path(data.get('dynamicData'))
    # print(f"code_path: {get_file_path(data.get('code'))}, ir_path: {get_file_path(data.get('ir'))}, cfg_path: {get_file_path(data.get('cfg'))}, dynamic_data_path: {get_file_path(data.get('dynamicData'))}")
    
    # 将内容存入字典
    contents = {
        'code': code_content,
        # 'ir': ir_content,
        # 'cfg': cfg_content
    }
    
       # 检查是否有有效输入
    if not any(contents.values()):
        return jsonify({"error": "No valid input provided"}), 400

    # 旧版本
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    #
    # async def async_task():
    #     # 当前仅处理 code，未来可扩展
    #     if contents['code']:
    #         future1 = loop.run_in_executor(None, analyze_hip, contents['code'])
    #         future2 = loop.run_in_executor(None, llm.improve_hip_code, contents['code'])
    #         analysis, improve = await asyncio.gather(future1, future2)
    #     else:
    #         # 如果没有 code，可根据 ir 或 cfg 扩展逻辑
    #         analysis = "No code provided"
    #         improve = "No code provided"
    #     return analysis, improve
    #
    # analysis, improve = loop.run_until_complete(async_task())
    # loop.close()

    if contents['code']:
        # 这里暂时不用大模型接口
        # improve = llm.improve_hip_code(contents['code'])
        improve = """
        // 优化后的核函数代码
__global__ void gauss_all_seidel_backfor(int mne, int nv, int* nc, double* a_ae, double* f,
                                         int* ne, double* ap, double* con, double* ff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mne)
    {
        double tmp_b = 0.0;
        int j_start = nc[i];
        int j_end = nc[i + 1];

        for (int j = j_start; j < j_end; ++j) // 使用局部变量简化索引访问
        {
            tmp_b += a_ae[j] * __ldg(&f[(nv - 1) * mne + ne[j] - 1]); // 使用__ldg提升全局内存读取速度
        }
        ff[i] = (tmp_b + con[i]) / ap[i];
    }
}
"""
        analysis = "Skipped analyze_hip per temporary requirement"
    else:
        analysis = "No code provided"
        improve = "No code provided"
    # 存入数据库,数据库内容还要修改，等后面确定了再说，4.28cjk；
    dcu_data = {
        'user_id': user_id,
        'file_id': None,  # 如果是文件上传，可在此记录 file_id
        'text': code_content,  # 暂时只存 code_content
        'analysis': analysis,
        'improve': improve
    }
    add_dcu(dcu_data)

    response_data = {
        'analysis': analysis,
        'improve': improve
    }
    return jsonify(response_data), 200


# DCU平台建模训练
@app.route('/dcu_model', methods=['POST'])
def handle_dcu_model():
    print("ni hao\n")
    data = request.json
    user_id = data['user_id']

    # 获取 code, ir, cfg 的内容
    code_content = get_content(data['code'])
    ir_content = get_content(data['ir'])
    cfg_content = get_content(data['cfg'])

    # 获取 code, ir, cfg, dynamicData 的文件路径
    code_path = get_file_path(data.get('code'))
    ir_path = get_file_path(data.get('ir'))
    cfg_path = get_file_path(data.get('cfg'))
    dynamic_data_path = get_file_path(data.get('dynamicData'))
    print(f"code_path: {code_path}, ir_path: {ir_path}, cfg_path: {cfg_path}, dynamic_data_path: {dynamic_data_path}")

    # 将内容存入字典
    contents = {
        'code': code_content,
        'ir': ir_content,
        'cfg': cfg_content
    }

    # 检查是否有有效输入
    if not any(contents.values()):
        return jsonify({"error": "No valid input provided"}), 400

        # 这是服务器上的地址 -> 修改为基于当前项目的相对路径
    base_path = os.getcwd()  # 或者使用 "." 代表当前目录
    code_path = os.path.join(base_path, code_path)
    ir_path = os.path.join(base_path, ir_path)
    dynamic_data_path = os.path.join(base_path, dynamic_data_path)
    cfg_path = os.path.join(base_path, cfg_path)

    # 构造命令
    command1 = [
        "python3", "/home/cjk/X-Blue/dataprocess.py",
        "--cpp", code_path,
        "--ll", ir_path,
        "--csv", dynamic_data_path,
        "--out_dir", "/home/cjk/X-Blue/output_embedding/",
        "--dot", cfg_path
    ]
    # 构造第二个命令
    command2 = [
        "python", "/home/cjk/X-Blue/modeling.py"
    ]
    # 执行命令
    # try:
    #     result = subprocess.run(command1, capture_output=True, text=True, check=True)
    #     print("命令执行成功")
    #     print("输出:", result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("命令执行失败")
    #     print("错误信息:", e.stderr)
    #     return jsonify({"error": "命令执行失败", "message": e.stderr}), 500

    # 执行第二个命令
    # try:
    #     result2 = subprocess.run(command2, capture_output=True, text=True, check=True)
    #     print("第二个命令执行成功")
    #     print("输出:", result2.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("第二个命令执行失败")
    #     print("错误信息:", e.stderr)
    #     return jsonify({"error": "第二个命令执行失败", "message": e.stderr}), 500

    # 存入数据库,数据库内容还要修改，等后面确定了再说，4.28cjk；
    dcu_data = {
        'user_id': user_id,
        'file_id': None,  # 如果是文件上传，可在此记录 file_id
        'text': code_content,  # 暂时只存 code_content
        'analysis': "单独建模功能",
        'improve': "单独建模功能"
    }
    add_dcu(dcu_data)

    # 🔧 新增：读取图片文件并转为 base64 编码
    # 修改：路径改为 hip_code_modeling/figure/...
    image_path = os.path.join("hip_code_modeling", "figure", "training_loss_curves.png")
    image_base64 = ""
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = "data:image/png;base64," + base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        image_base64 = ""

        # 修改：路径改为 hip_code_modeling/log/...
    txt_path = os.path.join("hip_code_modeling", "log", "training_log.txt")
    model_process = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            model_process = txt_file.read()  # 保留所有换行符
    except Exception as e:
        print(f"Error reading text file: {e}")
        model_process = ""

    response_data = {
        'model_process': model_process,
        'loss_image': image_base64  # 添加的字段
    }
    return jsonify(response_data), 200


# DCU平台建模评估
@app.route('/dcu_evaluate', methods=['POST'])
def handle_dcu_evaluate():
    data = request.json
    user_id = data['user_id']

    # 获取 code, ir, cfg 的内容
    code_content = ""
    # code_content = get_content(data['code'])
    # ir_content = get_content(data['ir'])
    # cfg_content = get_content(data['cfg'])

    # 获取 code, ir, cfg, dynamicData 的文件路径
    # code_path = get_file_path(data.get('code'))
    # ir_path = get_file_path(data.get('ir'))
    # cfg_path = get_file_path(data.get('cfg'))
    # dynamic_data_path = get_file_path(data.get('dynamicData'))
    # print(f"code_path: {get_file_path(data.get('code'))}, ir_path: {get_file_path(data.get('ir'))}, cfg_path: {get_file_path(data.get('cfg'))}, dynamic_data_path: {get_file_path(data.get('dynamicData'))}")

    # 将内容存入字典
    # contents = {
    #     'code': code_content,
    #     'ir': ir_content,
    #     'cfg': cfg_content
    # }

    # 检查是否有有效输入
    # if not any(contents.values()):
    #     return jsonify({"error": "No valid input provided"}), 400

    command = [
        "python", "/home/cjk/X-Blue/test_predict.py"
    ]
    # 执行命令
    # try:
    #     result = subprocess.run(command, capture_output=True, text=True, check=True)
    #     print("命令执行成功")
    #     print("输出:", result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("命令执行失败")
    #     print("错误信息:", e.stderr)
    #     return jsonify({"error": "命令执行失败", "message": e.stderr}), 500

        # 存入数据库,数据库内容还要修改，等后面确定了再说，4.28cjk；
    dcu_data = {
        'user_id': user_id,
        'file_id': None,  # 如果是文件上传，可在此记录 file_id
        'text': code_content,  # 暂时只存 code_content
        'analysis': "建模评估功能",
        'improve': "建模评估功能"
    }
    add_dcu(dcu_data)

    # 修改：路径改为 hip_code_modeling/figure/...
    image1_path = os.path.join("hip_code_modeling", "figure", "mae_test.png")
    image1_base64 = ""
    try:
        with open(image1_path, "rb") as image1_file:
            image1_base64 = "data:image/png;base64," + base64.b64encode(image1_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        image1_base64 = ""

        # 修改：路径改为 hip_code_modeling/figure/...
    image2_path = os.path.join("hip_code_modeling", "figure", "mape_test.png")
    image2_base64 = ""
    try:
        with open(image2_path, "rb") as image2_file:
            image2_base64 = "data:image/png;base64," + base64.b64encode(image2_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        image2_base64 = ""

        # 修改：路径改为 hip_code_modeling/figure/...
    image3_path = os.path.join("hip_code_modeling", "figure", "mse_test.png")
    image3_base64 = ""
    try:
        with open(image3_path, "rb") as image3_file:
            image3_base64 = "data:image/png;base64," + base64.b64encode(image3_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        image3_base64 = ""

    response_data = {
        'evaluate_image1': image1_base64,
        'evaluate_image2': image2_base64,
        'evaluate_image3': image3_base64
    }
    return jsonify(response_data), 200


# 迈创平台建模训练
@app.route('/mt_model', methods=['POST'])
def handle_mt_model():
    data = request.json
    user_id = data['user_id']

    # 获取 code, ir, cfg 的内容
    code_content = get_content(data['code'])
    ir_content = get_content(data['ir'])
    cfg_content = get_content(data['cfg'])

    # 获取 code, ir, cfg, dynamicData 的文件路径
    code_path = get_file_path(data.get('code'))
    ir_path = get_file_path(data.get('ir'))
    cfg_path = get_file_path(data.get('cfg'))
    dynamic_data_path = get_file_path(data.get('dynamicData'))
    print(f"code_path: {code_path}, ir_path: {ir_path}, cfg_path: {cfg_path}, dynamic_data_path: {dynamic_data_path}")

    # 将内容存入字典
    contents = {
        'code': code_content,
        'ir': ir_content,
        'cfg': cfg_content
    }

    # 检查是否有有效输入
    if not any(contents.values()):
        return jsonify({"error": "No valid input provided"}), 400

        # 修改为基于当前项目的相对路径
    base_path = os.getcwd()
    code_path = os.path.join(base_path, code_path)
    ir_path = os.path.join(base_path, ir_path)
    dynamic_data_path = os.path.join(base_path, dynamic_data_path)
    cfg_path = os.path.join(base_path, cfg_path)
    # 构造命令
    command1 = [
        "python3", "/home/cjk/DSP/dataprocess.py",
        "--cpp", code_path,
        "--ll", ir_path,
        "--csv", dynamic_data_path,
        "--out_dir", "/home/cjk/DSP/output_embedding/",
        "--dot", cfg_path
    ]
    # 构造第二个命令
    command2 = [
        "python", "/home/cjk/DSP/modeling.py"
    ]
    # 执行命令
    # try:
    #     result = subprocess.run(command1, capture_output=True, text=True, check=True)
    #     print("命令执行成功")
    #     print("输出:", result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("命令执行失败")
    #     print("错误信息:", e.stderr)
    #     return jsonify({"error": "命令执行失败", "message": e.stderr}), 500

        # 执行第二个命令
    # try:
    #     result2 = subprocess.run(command2, capture_output=True, text=True, check=True)
    #     print("第二个命令执行成功")
    #     print("输出:", result2.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("第二个命令执行失败")
    #     print("错误信息:", e.stderr)
    #     return jsonify({"error": "第二个命令执行失败", "message": e.stderr}), 500

        # 存入数据库,数据库内容还要修改，等后面确定了再说，4.28cjk；
    dcu_data = {
        'user_id': user_id,
        'file_id': None,  # 如果是文件上传，可在此记录 file_id
        'text': code_content,  # 暂时只存 code_content
        'analysis': "单独建模功能",
        'improve': "单独建模功能"
    }
    add_dcu(dcu_data)

    # 🔧 新增：读取图片文件并转为 base64 编码
    # 修改：路径改为 hip_code_modeling/figure/...
    image_path = os.path.join("hip_code_modeling", "figure", "training_loss_curves.png")
    image_base64 = ""
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = "data:image/png;base64," + base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        image_base64 = ""

        # 修改：路径改为 hip_code_modeling/log/...
    txt_path = os.path.join("hip_code_modeling", "log", "training_log.txt")
    model_process = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            model_process = txt_file.read()  # 保留所有换行符
    except Exception as e:
        print(f"Error reading text file: {e}")
        model_process = ""

    response_data = {
        'model_process': model_process,
        'loss_image': image_base64  # 添加的字段
    }
    return jsonify(response_data), 200


# 迈创平台建模评估
@app.route('/mt_evaluate', methods=['POST'])
def handle_mt_evaluate():
    data = request.json
    user_id = data['user_id']

    # 获取 code, ir, cfg 的内容
    code_content = ""
    command = [
        "python", "/home/cjk/DSP/test_predict.py"
    ]
    # 执行命令
    # try:
    #     result = subprocess.run(command, capture_output=True, text=True, check=True)
    #     print("命令执行成功")
    #     print("输出:", result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("命令执行失败")
    #     print("错误信息:", e.stderr)
    #     return jsonify({"error": "命令执行失败", "message": e.stderr}), 500

        # 存入数据库,数据库内容还要修改，等后面确定了再说，4.28cjk；
    dcu_data = {
        'user_id': user_id,
        'file_id': None,  # 如果是文件上传，可在此记录 file_id
        'text': code_content,  # 暂时只存 code_content
        'analysis': "建模评估功能",
        'improve': "建模评估功能"
    }
    add_dcu(dcu_data)

    # 修改：路径改为 hip_code_modeling/figure/...
    image1_path = os.path.join("hip_code_modeling", "figure", "mae_test.png")
    image1_base64 = ""
    try:
        with open(image1_path, "rb") as image1_file:
            image1_base64 = "data:image/png;base64," + base64.b64encode(image1_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        image1_base64 = ""

        # 修改：路径改为 hip_code_modeling/figure/...
    image2_path = os.path.join("hip_code_modeling", "figure", "mape_test.png")
    image2_base64 = ""
    try:
        with open(image2_path, "rb") as image2_file:
            image2_base64 = "data:image/png;base64," + base64.b64encode(image2_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        image2_base64 = ""

        # 修改：路径改为 hip_code_modeling/figure/...
    image3_path = os.path.join("hip_code_modeling", "figure", "mse_test.png")
    image3_base64 = ""
    try:
        with open(image3_path, "rb") as image3_file:
            image3_base64 = "data:image/png;base64," + base64.b64encode(image3_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        image3_base64 = ""

    response_data = {
        'evaluate_image1': image1_base64,
        'evaluate_image2': image2_base64,
        'evaluate_image3': image3_base64
    }
    return jsonify(response_data), 200


# 分析DCU平台代码（文件）,弃用
@app.route('/file_dcu_code', methods=['POST'])
def handle_dcu_file():
    data = request.json
    # 获取上传文件在服务器的位置并读取
    file_path = data.get('file_path')
    user_id = data.get('user_id')
    file_id = data.get('file_id')
    full_path = os.path.join(os.getcwd(), file_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 将lines列表转换为字符串
    input_code = ''.join(lines)  # 或者使用 '\n'.join(lines) 如果需要保留行间的换行符
    # 将文件传给大模型分析得到结果
    analysis = "建模分析结果"
    improve = "优化后代码"
    # analysis = analyze_hip(input_code)
    # improve= llm.improve_hip_code(input_code)
    # 存入数据库
    dcu_data = {
        'user_id': user_id,
        'file_id': file_id,
        'text': "",
        'analysis': analysis,
        'improve': improve
    }
    add_dcu(dcu_data)
    # 返回给前端
    response_data = {'analysis': analysis, 'improve': improve}
    return jsonify(response_data), 200



# 上传文件，需要进一步完善，规定可以接收的格式和数量，要考虑前端发送的文件可能同名
UPLOAD_FOLDER = 'upfile'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 检查文件类型
    allowed_extensions = ['.json', '.txt', '.c', '.cpp', '.cu', '.dot', '.ll', '.gimple', '.csv']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({'error': '文件格式不支持'}), 400

    # 生成 UUID
    unique_id = str(uuid.uuid4())

    # 构建新的文件名
    _, file_extension = os.path.splitext(file.filename)
    new_filename = f"{unique_id}{file_extension}"

    # 保存文件
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    file.save(file_path)

    # 上传的同名文件在数据库中不覆盖
    file_id = add_file({
        'filename': file.filename,
        'server_filename': new_filename
    })
    print("file_id:", file_id)
    return jsonify({'message': 'File uploaded successfully', 'filePath': file_path, 'file_id': file_id})


# 用户点了叉叉就会对应删除文件
@app.route('/delete-file', methods=['POST'])
def delete_file():
    data = request.json
    file_path = data['filePath']
    file_id = data['file_id']
    if not file_path:
        return jsonify({'error': 'File path not provided'}), 400
    full_path = os.path.join(os.getcwd(), file_path)
    try:
        if os.path.exists(full_path):
            server_filename = os.path.basename(full_path)
            print("删除：" + server_filename)
            os.remove(full_path)
            # delete_file_by_server_filename(server_filename)
            delete_file_by_file_id(file_id)
            return jsonify({'message': 'File deleted successfully'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f'Error deleting file: {e}')
        return jsonify({'error': 'Failed to delete file'}), 500




# 将表单数据转换为JSON文件
@app.route('/form-to-json', methods=['POST'])
def handle_form_to_json():
    try:
        # 获取前端传入的表单数据
        form_data = request.json
        if not form_data:
            return jsonify({'error': 'No form data provided'}), 400
        
        # 生成唯一的文件名
        unique_id = str(uuid.uuid4())
        json_filename = f"{unique_id}.json"
        json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        
        # 将表单数据写入JSON文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(form_data, json_file, ensure_ascii=False, indent=2)
        
        # 保存文件信息到数据库（可选）
        file_id = add_file({
            'filename': json_filename,  # 或者使用用户提供的文件名
            'server_filename': json_filename
        })
        
        # 返回成功响应
        return jsonify({
            'message': 'Form data converted to JSON file successfully',
            'filePath': json_file_path,
            'file_id': file_id,
            'filename': json_filename
        }), 200
        
    except Exception as e:
        print(f'Error converting form to JSON: {e}')
        return jsonify({'error': 'Failed to convert form to JSON file'}), 500


# 注册
@app.route("/register/", methods=['POST'])
def register():
    username = request.json.get("username")
    password = request.json.get("password")
    name = request.json.get("name")
    tel = request.json.get("tel")
    identity = request.json.get("identity")
    try:
        user = User.query.filter_by(username=username).first()
        if user:
            ret["meta"]["status"] = 500
            ret["meta"]["message"] = "该用户已注册"
        else:
            is_admin = identity == '1'
            user = User(username=username, password=password, name=name, tel=tel, is_admin=is_admin)
            ret["meta"]["status"] = 200
            ret["meta"]["message"] = "注册成功"
            db.session.add(user)
            db.session.commit()
        return jsonify(ret)
    except Exception as error:
        print(error)
        ret["meta"]["status"] = 500
        ret["meta"]["message"] = "后台程序出错"
        return jsonify(ret)

# 登录
@app.route("/login/", methods=['POST'])
def login():
    ret = {
        "data": {},
        "meta": {
            "status": 200,
            "message": ""
        }
    }
    # print(request.json)
    try:
        username = request.json["username"]
        password = request.json["password"]
        value = request.json["value"]
        user = User.query.filter_by(username=username, password=password)
        print(user.first())
        if not user:
            ret["meta"]["status"] = 500
            ret["meta"]["message"] = "用户不存在或密码错误"
            return jsonify(ret)
        elif user and user.first().password:
            dict = {
                "exp": int((datetime.now() + timedelta(days=1)).timestamp()),  # 过期时间
                "iat": int(datetime.now().timestamp()),  # 开始时间
                "id": user.first().id,
                "username": user.first().username,
            }
            token = jwt.encode(dict, SECRET_KEY, algorithm="HS256")
            ret["data"]["token"] = token
            ret["data"]["username"] = user.first().username
            ret["data"]["user_id"] = user.first().id
            ret["meta"]["status"] = 200
            ret["meta"]["message"] = "登录成功"
            # 前端发来的value为0用户登录，反之管理员登录
            if value == '0':
                # print("用户登录")
                ret["data"]["isAdmin"] = 0
            else:
                # 根据数据库判断是不是管理员
                if user.first().is_admin:
                    ret["data"]["isAdmin"] = 1
                else:
                    ret["data"]["isAdmin"] = 0
            # print(ret, type(ret))
            # user_data = verify_token(token)
            # print(user_data, type(user_data))
            return jsonify(ret)
        else:
            ret["meta"]["status"] = 500
            ret["meta"]["message"] = "用户不存在或密码错误"
            return jsonify(ret)
    except Exception as error:
        print(error)
        ret["meta"]["status"] = 500
        ret["meta"]["message"] = "用户不存在或密码错误"
        return jsonify(ret)
    

import os
import json
import re
import traceback
from flask import request, jsonify

def sanitize_filename(name: str, default: str = "unknown") -> str:

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

    base, ext = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{base}_{i}{ext}")
        i += 1
    return candidate

_SIZE_RE = re.compile(r"^(-?\d+)\s*([kKmMgGtT])?$")

def _parse_value_token(tok: str) -> str:

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

    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    parts = re.split(r"[,;\n\s]+", s)
    return [p.strip() for p in parts if p and p.strip()]

def _ensure_str_list(val):

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
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None

def normalize_payload_to_input_file_n(form_data: dict) -> dict:

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
from flask import request, jsonify, current_app
from api.mt3000_connect import Mt3000Client
@app.route('/io-collection', methods=['POST'])
def handle_io_collection():
    client = None

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
       

        # TODO：密码建议改成从环境变量或者配置文件读取，不要硬编码
        client = Mt3000Client(
            hostname="192.168.10.20",
            username="xjtu_cx",
            password="gCWyS6RmwEAT",
            proxy_host="127.0.0.1",
            proxy_port=1080,
        )
        remote_dir = f"/thfs3/home/{Mt3000Client.username}/MCJM-Toolkits/config/IOJM/"
        remote_json = os.path.join(remote_dir, server_filename)
        print(f"[IOJM] 开始上传 {file_path} 到远程 {remote_json} ...")
        client.upload_config(file_path, remote_json)
        print("[IOJM] 文件上传完成！")

        # 5. 在远程调用 IOJM main.py，执行对应步骤
        # 假设 main.py 路径为：/thfs3/home/xjtu_cx/MCJM-Toolkits/src/IOJM/main.py
        # -u 传配置文件路径
        # -s 传步骤：1=采集数据，2=数据处理，3=建模，4=预测
        remote_main = "/thfs3/home/xjtu_cx/MCJM-Toolkits/src/IOJM/main.py"
        cmd = (
            f"cd /thfs3/home/xjtu_cx/MCJM-Toolkits/src/IOJM && "
            f"python {remote_main} -u '{remote_json}' -s {step}"
        )
        print(f"[IOJM] 在远程执行命令: {cmd}")
        out, err = client.run_command(cmd)

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
        if client is not None:
            try:
                client.close()
                print("[IOJM] SSH 连接已关闭")
            except Exception as close_err:
                print("[IOJM] 关闭 SSH 连接时异常:", close_err)

# pucharm运行的时候好像不执行这个主函数，暂时没搞懂
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8009, debug=True)
