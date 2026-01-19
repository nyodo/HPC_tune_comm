from models import File, User, db
from flask import Blueprint, jsonify, request, render_template
from datetime import datetime

file_bp = Blueprint('file', __name__)

def paginate_list(data, page, per_page):
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = data[start:end]
    return paginated_data

@file_bp.route('/', methods=['GET'])
def get_all_file():
    # 获取搜索关键字
    per_page = request.args.get('pageSize', default=10, type=int)
    page = request.args.get('pageNum', default=1, type=int)
    search_query = request.args.get('query')
    if search_query:
        print(search_query)
        search_filter = db.or_(
            File.filename.ilike(f'%{search_query}%')
        )
        file_list = File.query.filter(search_filter).all()
        page_obj_zs = File.query.filter(search_filter).count()
    else:
        page_obj_zs = File.query.count()
        file_list = File.query.all()
    # 通过函数对 List 过滤
    file = paginate_list(file_list, page, per_page)

    result = [{'id': item.id, 'filename': item.filename, 'server_filename': item.server_filename,
               'created_time': item.created_time.strftime('%Y-%m-%d %H:%M:%S') if item.created_time else None} for item in file]
    print(result)
    data = {'code': 200, 'zs': page_obj_zs, 'data': result}
    return jsonify(data)

def add_file(file_data):
    # 获取当前时间
    now = datetime.now()
    # 将当前时间格式化为字符串
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file = File(
        filename=file_data['filename'],
        server_filename=file_data['server_filename'],
        created_time=current_time_str
    )
    db.session.add(file)
    db.session.commit()
    return file.id


def delete_file_by_server_filename(server_filename):
    file_to_delete = File.query.filter_by(server_filename=server_filename).first()
    db.session.delete(file_to_delete)
    db.session.commit()


def delete_file_by_file_id(file_id):
    file_to_delete = File.query.filter_by(id=file_id).first()
    db.session.delete(file_to_delete)
    db.session.commit()


# 根据 server_filename 名称，批量删除
def batch_delete_file_by_server_filename(server_filename_list):
    for server_filename in server_filename_list:
        if server_filename is not None:
            delete_file_by_server_filename(server_filename)


# 根据 file_id，批量删除
def batch_delete_file_by_file_id(file_id_list):
    for file_id in file_id_list:
        if file_id is not None:
            file_to_delete = File.query.filter_by(id=file_id).first()
            db.session.delete(file_to_delete)
    db.session.commit()


def get_filename_by_server_filename(server_filename):
    file_record = File.query.filter_by(server_filename=server_filename).first()
    if file_record:
        return file_record.filename
    else:
        return "文本"


def get_file_by_filename(filename):
    file = File.query.filter_by(filename=filename).first()
    if file:
        return file
    else:
        return None  # 或者返回一个默认值，例如空字典


def get_file_by_file_id(file_id):
    file = File.query.filter_by(id=file_id).first()
    if file:
        return file
    else:
        return None


def update_file_with_data(file_data):
    # 获取当前时间
    now = datetime.now()
    # 将当前时间格式化为字符串
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file = File.query.get(file_data['id'])
    file.server_filename = file_data['server_filename']
    file.created_time = current_time_str

    db.session.commit()
