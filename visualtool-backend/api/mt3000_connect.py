# -*- coding: utf-8 -*-
import paramiko
import socks  # 需安装 PySocks
from typing import Optional, Tuple


class Mt3000Client:
    """
    通过 SOCKS5 代理连接超算的通用客户端。

    方法一：`run_command` 直接执行命令并返回 stdout/stderr。
    方法二：`upload_config` 上传配置文件到超算指定路径。
    """

    def __init__(
        self,
        hostname: str,
        username: str,
        password: str,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 1080,
        ssh_port: int = 22,
        connect_timeout: int = 10,
    ) -> None:
        self.hostname = hostname
        self.username = username
        self.password = password
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.ssh_port = ssh_port
        self.connect_timeout = connect_timeout

        self._client: Optional[paramiko.SSHClient] = None
        self._proxy_sock: Optional[socks.socksocket] = None

    def connect(self) -> None:
        """建立通过 SOCKS5 代理的 SSH 连接。重复调用会复用已有连接。"""
        if self._client:
            return

        self._proxy_sock = socks.socksocket()
        self._proxy_sock.set_proxy(socks.SOCKS5, self.proxy_host, self.proxy_port)
        self._proxy_sock.connect((self.hostname, self.ssh_port))

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.hostname,
            username=self.username,
            password=self.password,
            sock=self._proxy_sock,
            timeout=self.connect_timeout,
        )

        self._client = client

    def run_command(self, command: str, timeout: int = 30, read_output: bool = True) -> Tuple[str, str]:
        """
        方法一：执行远程命令。
        返回 (stdout, stderr)，均为解码后的字符串。
        """
        self.connect()
        assert self._client is not None

        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)

        # 对于 nohup / 后台命令，不读取输出，避免 stdout.read() 阻塞/超时
        if not read_output:
            try:
                stdout.channel.close()
            except Exception:
                pass
            return "", ""

        return stdout.read().decode(), stderr.read().decode()

    def upload_config(self, local_path: str, remote_path: str) -> None:
        """
        方法二：上传配置文件到指定路径。
        local_path: 本地配置文件路径
        remote_path: 远端保存路径
        """
        self.connect()
        assert self._client is not None

        sftp = self._client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
        finally:
            sftp.close()

    def close(self) -> None:
        """关闭 SSH 及代理连接。"""
        if self._client:
            self._client.close()
            self._client = None
        if self._proxy_sock:
            self._proxy_sock.close()
            self._proxy_sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 示例用法：按需替换真实账户信息及路径。(放在配置文件里，目录)
if __name__ == "__main__":
    client = Mt3000Client(
        hostname="192.168.10.20",
        username="xjtu_cx",
        password="gCWyS6RmwEAT",
        proxy_host="127.0.0.1",
        proxy_port=1080,
    )
    try:
        out, err = client.run_command("ls")
        print("命令输出:\n", out)
        if err:
            print("命令错误:\n", err)
        # client.upload_config("local_config.sh", "/home/xjtu_cx/config.sh")
    finally:
        client.close()