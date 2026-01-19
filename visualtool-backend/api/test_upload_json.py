# -*- coding: utf-8 -*-
from mt3000_connect import Mt3000Client
import os
if __name__ == "__main__":
    client = Mt3000Client(
        hostname="192.168.10.20",
        username="xjtu_cx",
        password="gCWyS6RmwEAT",
        proxy_host="127.0.0.1",
        proxy_port=1080,
    )

    try:
            # 服务器上已有 JSON 文件路径
            local_json = "/home/cjk/visualtool-backend/api/test.json"
            if not os.path.exists(local_json):
                print(f"文件不存在: {local_json}")
                exit()

            # 远程保存路径，可选择放在同目录或者其他目录
            remote_json = f"/thfs3/home/{client.username}/liuheng/test.json"

            print(f"开始上传 {local_json} 到远程 {remote_json} ...")
            client.upload_config(local_json, remote_json)
            print("文件上传完成！")

            # 验证远程文件
            out, err = client.run_command(f"ls -lh {remote_json}")
            if out:
                print("远程文件信息:\n", out)
            if err:
                print("命令错误:\n", err)

            # 可选查看内容
            out, err = client.run_command(f"cat {remote_json}")
            if out:
                print("远程文件内容:\n", out)
            if err:
                print("命令错误:\n", err)

    except Exception as e:
        print("发生异常:", e)
    finally:
        client.close()
        print("SSH 连接已关闭")