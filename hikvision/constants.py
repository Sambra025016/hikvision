"""
hikvision.constants
~~~~~~~~~~~~~~~~~~~~

List of constants

Copyright (c) 2015 Finbarr Brady <https://github.com/fbradyirl>
Licensed under the MIT license.

本模块集中定义摄像机 PTZ 连接和 XML 请求使用的常量。
"""

# 未显式指定端口时，由 HTTP/HTTPS 协议选择默认端口。
DEFAULT_PORT = None
# 与摄像头交换 XML 数据时使用 UTF-8 编码。
XML_ENCODING = 'UTF-8'

# 摄像头 XML 接口所需的默认 HTTP 请求头。
DEFAULT_HEADERS = {
    'Content-Type': "application/xml; charset='UTF-8'",
    'Accept': "*/*"
}
