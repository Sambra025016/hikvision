"""完整运动目标跟踪链路的命令行入口。"""

import argparse
import getpass
import os
import sys
import time
from pathlib import Path

import cv2

from hikvision.ptz import PTZController
from motion_app.config import load_config
from motion_app.controller import HikvisionPTZExecutor
from motion_app.metrics import MetricsRecorder
from motion_app.pipeline import TrackingPipeline
from motion_app.sources import RTSPSource, SyntheticFrameSource, VideoFileSource
from motion_app.stream import build_hikvision_rtsp_url


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tracking.json"


def build_parser():
    parser = argparse.ArgumentParser(
        description="运动检测、卡尔曼跟踪、PTZ 决策与居中抓拍。"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="使用本地视频文件")
    source.add_argument(
        "--synthetic", action="store_true", help="使用内置合成视频"
    )
    source.add_argument(
        "--rtsp", action="store_true", help="连接海康威视 RTSP"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--host", default=os.getenv("HIKVISION_HOST"))
    parser.add_argument(
        "--username", default=os.getenv("HIKVISION_USERNAME")
    )
    parser.add_argument(
        "--password", default=os.getenv("HIKVISION_PASSWORD")
    )
    parser.add_argument("--rtsp-port", type=int, default=554)
    parser.add_argument("--ptz-port", type=int, default=None)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--stream", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--enable-ptz", action="store_true",
        help="实际发送云台命令；默认仅生成并记录命令",
    )
    parser.add_argument(
        "--compensation", choices=("identity", "orb"),
        help="临时覆盖配置中的运动补偿模式",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--synthetic-camera-shift", type=float, default=0.0,
        help="合成画面每帧背景水平位移像素，用于测试 ORB 补偿",
    )
    return parser


def _require_camera_credentials(parser, args):
    if not args.host:
        parser.error("RTSP 或真实 PTZ 模式需要 --host")
    if not args.username:
        parser.error("RTSP 或真实 PTZ 模式需要 --username")
    if args.password is None:
        args.password = getpass.getpass("摄像头密码：")


def _create_source(parser, args, config):
    if args.video:
        return VideoFileSource(args.video), "video"
    if args.rtsp:
        _require_camera_credentials(parser, args)
        url = build_hikvision_rtsp_url(
            args.host,
            args.username,
            args.password,
            port=args.rtsp_port,
            channel=args.channel,
            stream=args.stream,
        )
        return RTSPSource(url), "rtsp"
    size = config["preprocess"]
    return SyntheticFrameSource(
        width=size["width"],
        height=size["height"],
        camera_shift_per_frame=args.synthetic_camera_shift,
    ), "synthetic"


def _create_executor(parser, args):
    if not args.enable_ptz:
        return None
    _require_camera_credentials(parser, args)
    controller = PTZController(
        host=args.host,
        port=args.ptz_port,
        username=args.username,
        password=args.password,
        channel=args.channel,
    )
    return HikvisionPTZExecutor(controller)


def _open_video_writer(path, size, fps=15.0, color=True):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, color
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建输出视频：{}".format(path))
    return writer


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.compensation:
        config["compensation"]["mode"] = args.compensation
    if args.enable_ptz and not args.rtsp:
        parser.error("--enable-ptz 只能与 --rtsp 一起使用")

    source, source_name = _create_source(parser, args, config)
    executor = _create_executor(parser, args)
    output_root = PROJECT_ROOT / config["output"]["root"]
    metrics = MetricsRecorder(output_root)
    pipeline = TrackingPipeline(
        config, capture_dir=metrics.capture_dir, executor=executor
    )
    video_writer = None
    mask_writer = None
    processed_count = 0
    last_report = time.monotonic()

    print("输入源：{}；补偿模式：{}；PTZ：{}。".format(
        source_name,
        config["compensation"]["mode"],
        "真实执行" if args.enable_ptz else "dry-run",
    ))
    print("实验输出目录：{}".format(metrics.session_dir))
    if not args.no_show:
        print("按 q 或 ESC 退出。")

    try:
        while True:
            packet = source.read(timeout=1.0)
            if packet is None:
                if source_name == "rtsp":
                    continue
                break

            started = time.perf_counter()
            result = pipeline.process(packet)
            if result is None:
                continue
            processing_ms = (time.perf_counter() - started) * 1000.0
            metrics.record(result, processing_ms)
            processed_count += 1

            height, width = result.visualization.shape[:2]
            size = (width, height)
            if config["output"]["save_video"]:
                if video_writer is None:
                    fps = getattr(source, "fps", 15.0)
                    video_writer = _open_video_writer(
                        metrics.session_dir / "tracking.mp4", size, fps
                    )
                video_writer.write(result.visualization)
            if config["output"]["save_mask"]:
                if mask_writer is None:
                    fps = getattr(source, "fps", 15.0)
                    mask_writer = _open_video_writer(
                        metrics.session_dir / "motion_mask.mp4",
                        size,
                        fps,
                        color=False,
                    )
                mask_writer.write(result.detection.motion_mask)

            if not args.no_show:
                cv2.imshow("motion target tracking", result.visualization)
                cv2.imshow("motion mask", result.detection.motion_mask)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif time.monotonic() - last_report >= 1.0:
                print(
                    "frame={} targets={} track={} visible={} "
                    "position={} velocity=({:.1f},{:.1f}) PTZ=({},{})".format(
                        result.packet.frame_id,
                        len(result.detection.detections),
                        result.track.track_id,
                        bool(result.observation and result.observation.visible),
                        result.track.filtered_center,
                        result.track.velocity[0],
                        result.track.velocity[1],
                        result.command.pan,
                        result.command.tilt,
                    )
                )
                last_report = time.monotonic()

            if args.max_frames and processed_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n用户停止运行。")
    finally:
        pipeline.close()
        source.close()
        metrics.close()
        if video_writer is not None:
            video_writer.release()
        if mask_writer is not None:
            mask_writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print("处理完成：{} 帧。".format(processed_count))
    print("指标文件：{}".format(metrics.csv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
