from models import User, db
from flask import Blueprint, jsonify, request, render_template
from datetime import datetime

user_bp = Blueprint('user', __name__)


def paginate_list(data, page, per_page):
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = data[start:end]
    return paginated_data


@user_bp.route('/', methods=['GET'])
def get_all_user():
    page_obj_zs = User.query.count()
    user_list = User.query.all()
    # 获取搜索关键字
    per_page = request.args.get('pageSize', default=10, type=int)
    page = request.args.get('pageNum', default=1, type=int)
    search_query = request.args.get('query')
    if search_query:
        print(search_query)
        search_filter = db.or_(
            User.name.ilike(f'%{search_query}%'),
            User.tel.ilike(f'%{search_query}%'),
            User.username.ilike(f'%{search_query}%')
        )
        user_list = User.query.filter(search_filter).all()
        page_obj_zs = User.query.filter(search_filter).count()
    # 通过函数对 List 过滤
    user = paginate_list(user_list, page, per_page)

    result = [{'id': item.id, 'username': item.username, 'password': item.password,
               'tel': item.tel, 'name': item.name,
               'is_admin': item.is_admin,
               'created_time': item.created_time.strftime('%Y-%m-%d %H:%M:%S') if item.created_time else None} for item in user]
    print(result)
    data = {'code': 200, 'zs': page_obj_zs, 'data': result}
    return jsonify(data)


@user_bp.route('/', methods=['POST'])
def add_user():
    data = request.json
    # 获取当前时间
    now = datetime.now()
    # 将当前时间格式化为字符串
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    user = User(
        username=data['username'],
        password=data['password'],
        tel=data['tel'],
        name=data['name'],
        is_admin=data['is_admin'],
        created_time=current_time_str
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'code': 200, 'data': '增加成功'})


@user_bp.route('/<user_id>/', methods=['GET'])
def get_user(user_id):
    item = User.query.get(user_id)
    result = {'id': item.id, 'username': item.username, 'password': item.password,
              'tel': item.tel, 'name': item.name,
              'is_admin': item.is_admin, 'created_time': item.created_time}
    data = {'code': 200, 'data': result}
    return jsonify(data)


@user_bp.route('/<user_id>/', methods=['PUT'])
def update_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return 'User not found', 404

    data = request.json
    user.username = data['username']
    user.password = data['password']
    user.tel = data['tel']
    user.name = data['name']
    user.is_admin = data['is_admin']
    user.created_time = data['created_time']

    db.session.commit()
    return jsonify({'code': 200, 'data': '更新成功'})


@user_bp.route('/<user_id>/', methods=['DELETE'])
def delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return 'User not found', 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({'code': 200, 'data': '删除成功'})
