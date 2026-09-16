"""统一的视频文件、合成画面和 RTSP 输入适配器。"""

import time

import cv2
import numpy as np

from motion_app.models import FramePacket
from motion_app.stream import LatestFrameStream


class VideoFileSource:
    """按视频原始 FPS 生成稳定时间戳的离线输入源。"""

    def __init__(self, path):
        self.path = path
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise ValueError("无法打开视频文件：{}".format(path))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self._base_time = time.monotonic()
        self._frame_id = 0

    def read(self, timeout=None):
        del timeout
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        height, width = frame.shape[:2]
        packet = FramePacket(
            frame_id=self._frame_id,
            capture_time=self._base_time + self._frame_id / self.fps,
            bgr_frame=frame,
            source_width=width,
            source_height=height,
            stream_ok=True,
        )
        self._frame_id += 1
        return packet

    def close(self):
        self.capture.release()


class SyntheticFrameSource:
    """生成带纹理背景和移动目标的可重复测试画面。"""

    def __init__(self, width=640, height=360, fps=15.0,
                 duration_seconds=12.0, camera_shift_per_frame=0.0):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.total_frames = int(round(duration_seconds * fps))
        self.camera_shift_per_frame = float(camera_shift_per_frame)
        self._frame_id = 0
        self._base_time = time.monotonic()
        self._background = self._create_background()

    def _create_background(self):
        background = np.full(
            (self.height, self.width, 3), 35, dtype=np.uint8
        )
        for x in range(0, self.width, 40):
            cv2.line(background, (x, 0), (x, self.height), (55, 55, 55), 1)
        for y in range(0, self.height, 40):
            cv2.line(background, (0, y), (self.width, y), (55, 55, 55), 1)
        generator = np.random.RandomState(7)
        for _ in range(120):
            center = (
                int(generator.randint(0, self.width)),
                int(generator.randint(0, self.height)),
            )
            cv2.circle(background, center, 2, (90, 90, 90), -1)
        return background

    def read(self, timeout=None):
        del timeout
        if self._frame_id >= self.total_frames:
            return None
        shift = self._frame_id * self.camera_shift_per_frame
        transform = np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32)
        frame = cv2.warpAffine(
            self._background,
            transform,
            (self.width, self.height),
            borderMode=cv2.BORDER_REFLECT,
        )
        progress = self._frame_id / max(1.0, self.total_frames - 1)
        target_x = int(50 + progress * (self.width - 100))
        target_y = int(
            self.height / 2.0 + 55.0 * np.sin(progress * 4.0 * np.pi)
        )
        cv2.rectangle(
            frame,
            (target_x - 18, target_y - 38),
            (target_x + 18, target_y + 38),
            (230, 230, 230),
            -1,
        )
        cv2.circle(frame, (target_x, target_y - 48), 12, (230, 230, 230), -1)
        packet = FramePacket(
            frame_id=self._frame_id,
            capture_time=self._base_time + self._frame_id / self.fps,
            bgr_frame=frame,
            source_width=self.width,
            source_height=self.height,
            stream_ok=True,
        )
        self._frame_id += 1
        return packet

    def close(self):
        return None


class RTSPSource:
    """把异步最新帧采集器适配为统一的 read/close 接口。"""

    def __init__(self, rtsp_url, **stream_options):
        self.stream = LatestFrameStream(rtsp_url, **stream_options).start()
        self._last_frame_id = -1

    def read(self, timeout=1.0):
        packet = self.stream.read_latest(
            after_frame_id=self._last_frame_id, timeout=timeout
        )
        if packet is not None:
            self._last_frame_id = packet.frame_id
        return packet

    def close(self):
        self.stream.stop()

    def status(self):
        return self.stream.status()

