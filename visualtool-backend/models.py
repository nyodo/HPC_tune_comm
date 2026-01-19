from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    tel = db.Column(db.String(255))
    name = db.Column(db.String(255))
    is_admin = db.Column(db.Boolean, default=False)
    created_time = db.Column(db.DateTime, default=datetime.now)


class Dcu(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    file_id = db.Column(db.Integer)
    text = db.Column(db.Text, nullable=False)
    analysis = db.Column(db.Text, nullable=False)
    improve = db.Column(db.Text, nullable=False)
    created_time = db.Column(db.DateTime, default=datetime.now)
    # 定义与 User 表的关系
    user = db.relationship('User', backref=db.backref('dcus', lazy=True))


class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    server_filename = db.Column(db.String(255), nullable=False)
    created_time = db.Column(db.DateTime, default=datetime.now)