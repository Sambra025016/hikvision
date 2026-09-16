"""图像预处理模块。"""

from typing import Optional, Tuple

import cv2
import numpy as np

from motion_app.models import FramePacket, ProcessedFrame


class FramePreprocessor:
    """缩放、灰度化、降噪，并建立检测坐标到原图的映射。"""

    def __init__(self, target_size=(640, 360), blur_size=(5, 5), roi=None):
        self.target_size = tuple(target_size)
        self.blur_size = tuple(blur_size)
        self.roi = roi
        if len(self.target_size) != 2 or min(self.target_size) <= 0:
            raise ValueError("target_size 必须是正数 (width, height)")
        if any(value <= 0 or value % 2 == 0 for value in self.blur_size):
            raise ValueError("blur_size 必须由两个正奇数组成")

    def _valid_roi_mask(self, width, height):
        """把归一化 ROI 转为检测分辨率下的二值掩码。"""
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.roi is None:
            mask.fill(255)
            return mask

        if len(self.roi) != 4:
            raise ValueError("roi 必须是归一化的 (x, y, width, height)")
        x, y, roi_width, roi_height = self.roi
        if (x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0 or
                x + roi_width > 1 or y + roi_height > 1):
            raise ValueError("roi 各项必须在 0～1 内且不能超出画面")
        x1 = int(round(x * width))
        y1 = int(round(y * height))
        x2 = int(round((x + roi_width) * width))
        y2 = int(round((y + roi_height) * height))
        mask[y1:y2, x1:x2] = 255
        return mask

    def process(self, packet: FramePacket) -> Optional[ProcessedFrame]:
        """处理一个有效帧包；无效或空帧返回 None。"""
        frame = packet.bgr_frame
        if not packet.stream_ok or frame is None or frame.size == 0:
            return None
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None

        source_height, source_width = frame.shape[:2]
        target_width, target_height = self.target_size
        display = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_size, 0)
        valid_roi = self._valid_roi_mask(target_width, target_height)

        return ProcessedFrame(
            frame_id=packet.frame_id,
            capture_time=packet.capture_time,
            display_frame=display,
            gray_frame=gray,
            # 检测坐标乘此比例即可还原到原始码流坐标。
            scale_x=source_width / float(target_width),
            scale_y=source_height / float(target_height),
            valid_roi=valid_roi,
        )

