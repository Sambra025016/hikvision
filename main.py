#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""海康威视球机 PTZ 自检与手动控制入口。"""

import argparse
import getpass
import os
import sys
import time

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from hikvision.error import HikvisionError
from hikvision.ptz import PTZController, build_url_base


PTZ_ACTIONS = (
    "left", "right", "up", "down", "zoom-in", "zoom-out", "stop"
)


def env_flag(name):
    """读取环境变量中的布尔值。"""
    return os.getenv(name, "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def env_int(name, default=None):
    """读取可选的整数环境变量。"""
    value = os.getenv(name)
    return default if not value else int(value)


def build_parser():
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="检查或手动控制海康威视球机 PTZ。"
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="ptz-check",
        choices=("diagnose", "ptz-check") + PTZ_ACTIONS,
        help="操作类型，默认执行只读 PTZ 能力检查",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HIKVISION_HOST"),
        help="摄像机 IP 或域名，也可设置 HIKVISION_HOST",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=env_int("HIKVISION_PORT"),
        help="摄像机 HTTP 端口，也可设置 HIKVISION_PORT",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("HIKVISION_USERNAME"),
        help="摄像机用户名，也可设置 HIKVISION_USERNAME",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("HIKVISION_PASSWORD"),
        help="摄像机密码；建议使用环境变量或隐藏输入",
    )
    parser.add_argument(
        "--https",
        action="store_true",
        default=env_flag("HIKVISION_HTTPS"),
        help="使用 HTTPS",
    )
    parser.add_argument(
        "--basic-auth",
        action="store_true",
        default=env_flag("HIKVISION_BASIC_AUTH"),
        help="使用 Basic 认证；默认使用 Digest",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=env_int("HIKVISION_CHANNEL", 1),
        help="PTZ 通道号，直连摄像机通常为 1",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=10,
        help="PTZ 速度，范围 1～100，默认 10",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.3,
        help="PTZ 运动时间，范围 0.1～3.0 秒，默认 0.3",
    )
    return parser


def require_connection_args(parser, args):
    """检查连接参数并安全读取密码。"""
    if not args.host:
        parser.error("请通过 --host 或 HIKVISION_HOST 指定摄像机地址")
    if not args.username:
        parser.error("请通过 --username 或 HIKVISION_USERNAME 指定用户名")
    if args.password is None:
        args.password = getpass.getpass("摄像机密码：")


def create_ptz_controller(args):
    """根据命令行连接参数创建 PTZ 控制器。"""
    return PTZController(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        is_https=args.https,
        digest_auth=not args.basic_auth,
        channel=args.channel,
    )


def diagnose(args):
    """验证 ISAPI 地址和当前 HTTP 认证信息。"""
    url = build_url_base(args.host, args.port, args.https)
    url += "/ISAPI/System/deviceInfo"
    auth_class = HTTPBasicAuth if args.basic_auth else HTTPDigestAuth
    response = requests.get(
        url,
        auth=auth_class(args.username, args.password),
        timeout=10,
    )
    print("诊断地址：{}".format(url))
    print("最终 HTTP 状态码：{}".format(response.status_code))
    if response.status_code == 200:
        print("诊断结果：ISAPI 已开启，当前账号认证成功。")
        return 0
    if response.status_code == 401:
        raise HikvisionError("ISAPI 认证失败，请检查账号和认证模式。")
    if response.status_code == 403:
        raise HikvisionError("设备拒绝访问，请检查当前账号权限。")
    if response.status_code == 404:
        raise HikvisionError("设备不存在此 ISAPI 接口。")
    raise HikvisionError("设备返回 HTTP %s。" % response.status_code)


def check_ptz(args):
    """只读查询并显示连续 PTZ 能力。"""
    capabilities = create_ptz_controller(args).get_capabilities()
    print("连续水平速度范围：{}".format(capabilities["pan"]))
    print("连续垂直速度范围：{}".format(capabilities["tilt"]))
    print("连续变焦速度范围：{}".format(capabilities["zoom"]))
    supported = (
        not capabilities["continuous_not_supported"] and
        capabilities["pan"] is not None and
        capabilities["tilt"] is not None
    )
    if not supported:
        raise HikvisionError("设备未声明连续云台运动能力。")
    print("检测结果：设备支持连续云台运动控制。")
    return 0


def run_ptz_action(args):
    """执行限时 PTZ 动作并保证最后发送停止命令。"""
    if not 1 <= args.speed <= 100:
        raise ValueError("--speed 必须在 1 到 100 之间")
    if not 0.1 <= args.duration <= 3.0:
        raise ValueError("--duration 必须在 0.1 到 3.0 秒之间")

    vectors = {
        "left": (-args.speed, 0, 0),
        "right": (args.speed, 0, 0),
        "up": (0, args.speed, 0),
        "down": (0, -args.speed, 0),
        "zoom-in": (0, 0, args.speed),
        "zoom-out": (0, 0, -args.speed),
    }
    controller = create_ptz_controller(args)
    capabilities = controller.get_capabilities()
    if capabilities["continuous_not_supported"]:
        raise HikvisionError("设备不支持连续 PTZ 控制。")

    if args.action == "stop":
        controller.stop()
        print("已发送云台停止命令。")
        return 0

    pan, tilt, zoom = vectors[args.action]
    print("执行动作：{}，速度 {}，持续 {} 秒。".format(
        args.action, args.speed, args.duration))
    try:
        controller.continuous_move(pan=pan, tilt=tilt, zoom=zoom)
        time.sleep(args.duration)
    finally:
        controller.stop()
    print("动作完成，已发送停止命令。")
    return 0


def main():
    """解析参数并执行设备自检或 PTZ 操作。"""
    parser = build_parser()
    try:
        args = parser.parse_args()
        require_connection_args(parser, args)
        if args.action == "diagnose":
            return diagnose(args)
        if args.action == "ptz-check":
            return check_ptz(args)
        return run_ptz_action(args)
    except (HikvisionError, requests.RequestException, ValueError) as exc:
        print("操作失败：{}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n操作已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
