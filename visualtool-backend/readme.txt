文件夹介绍：
1.api 数据库操作的路由和函数
2.static 静态文件如图片、文本
3.upfile前端上传到后端服务器的文件
4.migrations数据库迁移用到的文件夹，如果没有这个文件夹，终端执行flask db init
4.app.py 项目启动文件 models.py数据库表配置（里面是之前的一些表，后面会改） requ.txt需要安装库 settings.py项目配置文件

项目搭建流程：
1.先创建一个空的mysql数据库。app.py中SQLALCHEMY_DATABASE_URI变量改成本机的数据库路径。
2.迁移数据库：flask db init(第一次初始化) flask db migrate（生成迁移脚本） flask db upgrade（运行脚本，数据库创建对应表）
3.运行app.py