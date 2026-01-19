import json

from flask import Blueprint, jsonify, request, render_template
from datetime import datetime

from models import Dcu, db

dcu_bp = Blueprint('dcu', __name__)


def paginate_list(data, page, per_page):
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = data[start:end]
    return paginated_data


@dcu_bp.route('/', methods=['GET'])
def get_all_dcu():
    page_obj_zs = Dcu.query.count()
    dcu_list = Dcu.query.all()
    # 获取搜索关键字，这里可以增加获取用户id，返回用户自己的记录
    per_page = request.args.get('pageSize', default=10, type=int)
    page = request.args.get('pageNum', default=1, type=int)
    search_query = request.args.get('query')
    if search_query:
        print(search_query)
        search_filter = db.or_(
            Dcu.text.ilike(f'%{search_query}%'),
        )
        dcu_list = Dcu.query.filter(search_filter).all()
        page_obj_zs = Dcu.query.filter(search_filter).count()
    # 通过函数对 List 过滤
    dcu = paginate_list(dcu_list, page, per_page)

    result = [{'id': item.id, 'user_id': item.user_id, 'text': item.text, 'analysis': item.analysis, 'improve': item.improve,
               'created_time': item.created_time.strftime('%Y-%m-%d %H:%M:%S') if item.created_time else None} for item in dcu]
    print(result)
    data = {'code': 200, 'zs': page_obj_zs, 'data': result}
    return jsonify(data)


def add_dcu(dcu_data):
    # 获取当前时间
    now = datetime.now()
    # 将当前时间格式化为字符串
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    dcu = Dcu(
        user_id=dcu_data['user_id'],
        text=dcu_data['text'],
        analysis=dcu_data['analysis'],
        improve=dcu_data['improve'],
        created_time=current_time_str
    )
    db.session.add(dcu)
    db.session.commit()
    # return jsonify({'code': 200, 'data': '增加成功'})


def get_dcu_by_text(text):
    dcu = Dcu.query.filter_by(text=text).first()
    if dcu:
        return dcu
    else:
        return None  # 或者返回一个默认值，例如空字典


def update_dcu_with_data(dcu_data):
    # 获取当前时间
    now = datetime.now()
    # 将当前时间格式化为字符串
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    dcu = Dcu.query.get(dcu_data['id'])
    dcu.analysis = dcu_data['analysis']
    dcu.improve = dcu_data['improve']
    dcu.created_time = current_time_str

    db.session.commit()


@dcu_bp.route('/<dcu_id>/', methods=['GET'])
def get_dcu(dcu_id):
    item = Dcu.query.get(dcu_id)
    result = {'id': item.id, 'user_id': item.user_id, 'text': item.text, 'analysis': item.analysis, 'improve': item.improve, 'created_time': item.created_time}
    data = {'code': 200, 'data': result}
    return jsonify(data)


@dcu_bp.route('/<dcu_id>/', methods=['PUT'])
def update_dcu(dcu_id):
    dcu = Dcu.query.get(dcu_id)
    if not dcu:
        return 'Dcu not found', 404

    data = request.json
    dcu.user_id = data['user_id']
    dcu.text = data['text']
    dcu.analysis = data['analysis']
    dcu.improve = data['improve']
    dcu.created_time = data['created_time']

    db.session.commit()
    return jsonify({'code': 200, 'data': '更新成功'})


@dcu_bp.route('/<dcu_id>/', methods=['DELETE'])
def delete_dcu(dcu_id):
    dcu = Dcu.query.get(dcu_id)
    if not dcu:
        return 'Dcu not found', 404

    db.session.delete(dcu)
    db.session.commit()
    return jsonify({'code': 200, 'data': '删除成功'})
