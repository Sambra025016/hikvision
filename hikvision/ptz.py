"""
hikvision.ptz
~~~~~~~~~~~~~~~~~~~~

海康威视摄像机云台能力查询和连续运动控制。
"""

from xml.etree import ElementTree

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from hikvision.constants import DEFAULT_HEADERS, DEFAULT_PORT, XML_ENCODING
from hikvision.error import HikvisionError


def build_url_base(host, port, is_https):
    """根据连接配置构造 HTTP 或 HTTPS 基础地址。"""
    scheme = "https" if is_https else "http"
    base = "%s://%s" % (scheme, host)
    if port:
        base += ":%s" % port
    return base


def tree_no_ns_from_string(response):
    """移除默认 XML 命名空间并解析元素树。"""
    import re
    text = re.sub(' xmlns="[^"]+"', '', response)
    return ElementTree.fromstring(text)


class PTZController:
    """通过 ISAPI 控制单个摄像机通道的云台。"""

    def __init__(self, host, username, password, port=DEFAULT_PORT,
                 is_https=False, digest_auth=True, channel=1, timeout=10):
        self._host = host
        self._username = username
        self._password = password
        self._channel = channel
        self._timeout = timeout
        self._auth_fn = HTTPDigestAuth if digest_auth else HTTPBasicAuth
        self._base = build_url_base(host, port, is_https)
        self._ptz_url = "%s/ISAPI/PTZCtrl/channels/%s" % (
            self._base, self._channel)
        # 复用同一 HTTP 会话和认证对象，使 Digest nonce 计数保持连续。
        self._session = requests.Session()
        self._session.auth = self._auth_fn(self._username, self._password)

    def _check_response(self, response, operation):
        """把常见 HTTP 错误转换为便于理解的设备错误。"""
        if response.status_code == 401:
            attempts = list(response.history) + [response]
            challenges = [
                item.headers.get("WWW-Authenticate")
                for item in attempts
                if item.headers.get("WWW-Authenticate")
            ]
            detail = challenges[-1] if challenges else "设备未返回认证挑战"
            raise HikvisionError(
                "云台%s失败：摄像机认证失败；认证挑战：%s" %
                (operation, detail))
        if response.status_code == 403:
            raise HikvisionError("云台%s失败：当前账号没有 PTZ 权限。" % operation)
        if response.status_code == 404:
            raise HikvisionError(
                "云台%s失败：通道 %s 没有对应的 PTZ 接口。" %
                (operation, self._channel))
        if response.status_code != 200:
            raise HikvisionError(
                "云台%s失败：HTTP %s，响应：%s" %
                (operation, response.status_code, response.text))

    @staticmethod
    def _read_range(tree, space_name, axis_name):
        """从能力 XML 中读取指定运动轴的最小值和最大值。"""
        minimum = tree.find(
            ".//%s/%sRange/Min" % (space_name, axis_name))
        maximum = tree.find(
            ".//%s/%sRange/Max" % (space_name, axis_name))
        if minimum is None or maximum is None:
            return None
        return int(float(minimum.text)), int(float(maximum.text))

    def get_capabilities(self):
        """查询通道能力，并返回连续移动和变焦的取值范围。"""
        response = self._session.get(
            self._ptz_url + "/capabilities",
            timeout=self._timeout,
        )
        self._check_response(response, "能力查询")
        try:
            tree = tree_no_ns_from_string(response.text)
        except ElementTree.ParseError as exc:
            raise HikvisionError("无法解析云台能力 XML。", exc)

        unsupported = tree.find(".//notSupportPTZContinuous")
        return {
            "continuous_not_supported": (
                unsupported is not None and
                unsupported.text.strip().lower() == "true"
            ),
            "pan": self._read_range(
                tree, "ContinuousPanTiltSpace", "X"),
            "tilt": self._read_range(
                tree, "ContinuousPanTiltSpace", "Y"),
            "zoom": self._read_range(
                tree, "ContinuousZoomSpace", "Z"),
        }

    def continuous_move(self, pan=0, tilt=0, zoom=0):
        """按给定速度持续运动；三个参数的有效范围均为 -100～100。"""
        values = {"pan": pan, "tilt": tilt, "zoom": zoom}
        for name, value in values.items():
            if not isinstance(value, int) or not -100 <= value <= 100:
                raise ValueError("%s 必须是 -100 到 100 之间的整数" % name)

        root = ElementTree.Element(
            "PTZData",
            {
                "version": "2.0",
                "xmlns": "http://www.hikvision.com/ver20/XMLSchema",
            },
        )
        for name in ("pan", "tilt", "zoom"):
            ElementTree.SubElement(root, name).text = str(values[name])
        # Python 3.7 的 tostring 尚不支持 xml_declaration 参数，手动补充声明。
        payload = b'<?xml version="1.0" encoding="UTF-8"?>' + \
            ElementTree.tostring(root, encoding=XML_ENCODING)

        # 复制默认请求头，避免修改其他请求共享的全局字典。
        headers = DEFAULT_HEADERS.copy()
        headers["Content-Length"] = str(len(payload))
        headers["Host"] = self._host
        response = self._session.put(
            self._ptz_url + "/continuous",
            data=payload,
            headers=headers,
            timeout=self._timeout,
        )
        self._check_response(response, "连续运动控制")
        return response.text

    def stop(self):
        """将三个运动轴的速度全部置零，使云台停止。"""
        return self.continuous_move(pan=0, tilt=0, zoom=0)
