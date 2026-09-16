"""低延迟 RTSP 最新帧采集模块。"""

import threading
import time
from collections import deque
from typing import Optional
from urllib.parse import quote

import cv2

from motion_app.models import FramePacket, StreamStatus


def build_hikvision_rtsp_url(host, username, password, port=554,
                             channel=1, stream=2):
    """构造海康威视 RTSP 地址；stream=1 为主码流，2 为子码流。"""
    if not host or not username or password is None:
        raise ValueError("host、username 和 password 不能为空")
    if channel < 1 or stream not in (1, 2):
        raise ValueError("channel 必须大于 0，stream 只能是 1 或 2")
    channel_code = channel * 100 + stream
    user = quote(str(username), safe="")
    secret = quote(str(password), safe="")
    return "rtsp://{}:{}@{}:{}/Streaming/Channels/{}".format(
        user, secret, host, port, channel_code
    )


class LatestFrameStream:
    """在独立线程中采集 RTSP，并始终只保留最新帧。"""

    def __init__(self, rtsp_url, failure_limit=5, reconnect_delay=1.0,
                 reconnect_delay_max=8.0):
        self._rtsp_url = rtsp_url
        self._failure_limit = max(1, int(failure_limit))
        self._reconnect_delay = max(0.1, float(reconnect_delay))
        self._reconnect_delay_max = max(
            self._reconnect_delay, float(reconnect_delay_max)
        )
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = None
        self._capture = None
        self._latest = None
        self._last_delivered_id = -1
        self._next_frame_id = 0
        self._connected = False
        self._frames_received = 0
        self._dropped_by_overwrite = 0
        self._read_failures = 0
        self._reconnect_count = 0
        self._last_frame_time = None
        self._last_error = ""
        self._recent_times = deque(maxlen=30)

    def start(self):
        """启动采集线程；重复调用不会创建第二个线程。"""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="rtsp-capture", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout=3.0):
        """停止线程并释放 VideoCapture。"""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))
        capture = self._capture
        if capture is not None:
            capture.release()
        self._capture = None
        self._connected = False

    def read_latest(self, after_frame_id=None, timeout=None):
        """等待并返回比指定帧号更新的帧；超时或停止时返回 None。"""
        minimum_id = -1 if after_frame_id is None else int(after_frame_id)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while (self._latest is None or
                   self._latest.frame_id <= minimum_id):
                if self._stop_event.is_set():
                    return None
                wait_time = None
                if deadline is not None:
                    wait_time = deadline - time.monotonic()
                    if wait_time <= 0:
                        return None
                self._condition.wait(wait_time)
            packet = self._latest
            self._last_delivered_id = max(
                self._last_delivered_id, packet.frame_id
            )
            return packet

    def status(self):
        """返回当前连接及统计状态的快照。"""
        with self._condition:
            measured_fps = 0.0
            if len(self._recent_times) >= 2:
                elapsed = self._recent_times[-1] - self._recent_times[0]
                if elapsed > 0:
                    measured_fps = (len(self._recent_times) - 1) / elapsed
            return StreamStatus(
                connected=self._connected,
                frames_received=self._frames_received,
                dropped_by_overwrite=self._dropped_by_overwrite,
                read_failures=self._read_failures,
                reconnect_count=self._reconnect_count,
                measured_fps=measured_fps,
                last_frame_time=self._last_frame_time,
                last_error=self._last_error,
            )

    def _open_capture(self):
        """创建低缓冲 VideoCapture；错误信息中不包含带密码的 URL。"""
        capture = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        opened = capture.open(self._rtsp_url, cv2.CAP_FFMPEG)
        if not opened:
            capture.release()
            return None
        return capture

    def _publish(self, frame, stream_ok):
        now = time.monotonic()
        height, width = frame.shape[:2] if frame is not None else (0, 0)
        packet = FramePacket(
            frame_id=self._next_frame_id,
            capture_time=now,
            bgr_frame=frame,
            source_width=width,
            source_height=height,
            stream_ok=stream_ok,
        )
        self._next_frame_id += 1
        with self._condition:
            if (self._latest is not None and
                    self._latest.frame_id > self._last_delivered_id):
                self._dropped_by_overwrite += 1
            self._latest = packet
            self._condition.notify_all()

    def _run(self):
        delay = self._reconnect_delay
        first_attempt = True
        while not self._stop_event.is_set():
            capture = self._open_capture()
            self._capture = capture
            if capture is None:
                with self._condition:
                    self._connected = False
                    self._last_error = "RTSP 连接失败"
                    if not first_attempt:
                        self._reconnect_count += 1
                self._publish(None, False)
                first_attempt = False
                self._stop_event.wait(delay)
                delay = min(delay * 2, self._reconnect_delay_max)
                continue

            with self._condition:
                self._connected = True
                self._last_error = ""
                if not first_attempt:
                    self._reconnect_count += 1
            first_attempt = False
            delay = self._reconnect_delay
            consecutive_failures = 0

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    with self._condition:
                        self._read_failures += 1
                    if consecutive_failures >= self._failure_limit:
                        break
                    continue

                consecutive_failures = 0
                now = time.monotonic()
                with self._condition:
                    self._connected = True
                    self._frames_received += 1
                    self._last_frame_time = now
                    self._recent_times.append(now)
                self._publish(frame, True)

            capture.release()
            self._capture = None
            with self._condition:
                self._connected = False
                self._last_error = "RTSP 连续读取失败，准备重连"
            self._publish(None, False)
            if not self._stop_event.is_set():
                self._stop_event.wait(delay)
                delay = min(delay * 2, self._reconnect_delay_max)

        with self._condition:
            self._connected = False
            self._condition.notify_all()

