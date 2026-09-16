"""两帧差分运动检测模块。

检测器接收统一的 CompensationOutput。当前阶段由 IdentityFramePairBuilder
提供未补偿的相邻帧；以后接入全局运动补偿模块时，无需改变检测器接口。
"""

import time
from typing import Optional

import cv2
import numpy as np

from motion_app.models import (
    CompensationOutput,
    Detection,
    DetectionResult,
    DifferenceQuality,
    ProcessedFrame,
)


class IdentityFramePairBuilder:
    """把连续预处理帧封装为两帧差分输入，不执行空间对齐。

    仅适合摄像机静止阶段。球机开始运动后，应替换为真正的全局运动补偿模块。
    """

    def __init__(self):
        self._previous = None

    def reset(self):
        """断流、画面尺寸变化或云台运动后清空历史帧。"""
        self._previous = None

    def build(self, current: ProcessedFrame) -> CompensationOutput:
        """生成与后续运动补偿模块一致的输出对象。"""
        previous = self._previous
        self._previous = current
        height, width = current.gray_frame.shape[:2]

        if (previous is None or
                previous.gray_frame.shape != current.gray_frame.shape):
            return CompensationOutput(
                frame_id=current.frame_id,
                capture_time=current.capture_time,
                display_frame=current.display_frame,
                current_gray=current.gray_frame,
                aligned_previous=None,
                aligned_old=None,
                valid_mask=current.valid_roi.copy(),
                transform_previous=None,
                transform_old=None,
                quality={"mode": "identity", "previous_frame_id": None},
                compensation_ok=False,
                reason="waiting_for_previous_frame",
            )

        valid_mask = cv2.bitwise_and(
            current.valid_roi, previous.valid_roi
        )
        identity = np.eye(3, dtype=np.float32)
        return CompensationOutput(
            frame_id=current.frame_id,
            capture_time=current.capture_time,
            display_frame=current.display_frame,
            current_gray=current.gray_frame,
            aligned_previous=previous.gray_frame,
            aligned_old=None,
            valid_mask=valid_mask,
            transform_previous=identity,
            transform_old=None,
            quality={
                "mode": "identity",
                "previous_frame_id": previous.frame_id,
                "previous_capture_time": previous.capture_time,
                "overlap_ratio": float(
                    cv2.countNonZero(valid_mask)
                ) / float(width * height),
            },
            compensation_ok=True,
            reason="",
        )


class MotionDetector:
    """从已对齐的相邻灰度帧中提取结构化运动候选。"""

    def __init__(self, diff_thresh=25, min_area_ratio=0.002,
                 min_area=150, max_bbox_ratio=0.35, min_fill=0.12,
                 min_aspect=0.1, max_aspect=4.0,
                 open_kernel=(3, 3), close_kernel=(9, 5),
                 dilate_iterations=1):
        if not 0 <= diff_thresh <= 255:
            raise ValueError("diff_thresh 必须在 0～255 之间")
        self.diff_thresh = int(diff_thresh)
        self.min_area_ratio = float(min_area_ratio)
        self.min_area = int(min_area)
        self.max_bbox_ratio = float(max_bbox_ratio)
        self.min_fill = float(min_fill)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)
        self.dilate_iterations = max(0, int(dilate_iterations))
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(open_kernel)
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(close_kernel)
        )

    @staticmethod
    def _empty_result(data, reason):
        """构造检测不可用结果，与“正常但没有目标”明确区分。"""
        mask = np.zeros_like(data.current_gray, dtype=np.uint8)
        quality = DifferenceQuality(
            threshold=0,
            changed_pixel_ratio=0.0,
            processing_time_ms=0.0,
            previous_frame_id=data.quality.get("previous_frame_id"),
            frame_interval=None,
        )
        return DetectionResult(
            source_frame_id=data.frame_id,
            motion_mask=mask,
            detections=tuple(),
            quality=quality,
            detection_ok=False,
            reason=reason,
        )

    def detect(self, data: CompensationOutput) -> DetectionResult:
        """执行两帧差分并输出 mask、Detection 列表和质量信息。"""
        started = time.perf_counter()
        if not data.compensation_ok or data.aligned_previous is None:
            return self._empty_result(
                data, data.reason or "compensation_unavailable"
            )
        if data.current_gray.shape != data.aligned_previous.shape:
            return self._empty_result(data, "frame_size_changed")
        if data.valid_mask.shape != data.current_gray.shape:
            return self._empty_result(data, "invalid_mask_shape")

        raw_difference = cv2.absdiff(
            data.current_gray, data.aligned_previous
        )
        _, motion_mask = cv2.threshold(
            raw_difference, self.diff_thresh, 255, cv2.THRESH_BINARY
        )
        motion_mask = cv2.bitwise_and(motion_mask, data.valid_mask)
        motion_mask = cv2.morphologyEx(
            motion_mask, cv2.MORPH_OPEN, self.open_kernel
        )
        motion_mask = cv2.morphologyEx(
            motion_mask, cv2.MORPH_CLOSE, self.close_kernel
        )
        if self.dilate_iterations:
            motion_mask = cv2.dilate(
                motion_mask,
                self.close_kernel,
                iterations=self.dilate_iterations,
            )

        contours, _ = cv2.findContours(
            motion_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = motion_mask.shape
        frame_area = width * height
        area_threshold = max(
            self.min_area, int(frame_area * self.min_area_ratio)
        )
        detections = []

        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < area_threshold:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            bbox_area = int(box_width * box_height)
            fill_ratio = contour_area / float(bbox_area)
            aspect = box_width / float(box_height)
            if bbox_area > frame_area * self.max_bbox_ratio:
                continue
            if fill_ratio < self.min_fill:
                continue
            if not self.min_aspect <= aspect <= self.max_aspect:
                continue

            foreground = motion_mask[y:y + box_height, x:x + box_width]
            motion_energy = (
                cv2.countNonZero(foreground) / float(bbox_area)
            )
            detections.append(Detection(
                bbox=(x, y, box_width, box_height),
                center=(x + box_width / 2.0, y + box_height / 2.0),
                contour_area=contour_area,
                bbox_area=bbox_area,
                fill_ratio=fill_ratio,
                motion_energy=motion_energy,
                source_frame_id=data.frame_id,
            ))

        detections.sort(key=lambda item: item.bbox_area, reverse=True)
        valid_pixels = max(1, cv2.countNonZero(data.valid_mask))
        previous_time = data.quality.get("previous_capture_time")
        frame_interval: Optional[float] = None
        if previous_time is not None:
            frame_interval = data.capture_time - float(previous_time)
        quality = DifferenceQuality(
            threshold=self.diff_thresh,
            changed_pixel_ratio=(
                cv2.countNonZero(motion_mask) / float(valid_pixels)
            ),
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            previous_frame_id=data.quality.get("previous_frame_id"),
            frame_interval=frame_interval,
        )
        return DetectionResult(
            source_frame_id=data.frame_id,
            motion_mask=motion_mask,
            detections=tuple(detections),
            quality=quality,
            detection_ok=True,
            reason="",
        )


def draw_result(frame, result: DetectionResult):
    """在显示帧副本上绘制结构化 Detection；第一个框为最大候选。"""
    visualization = frame.copy()
    for index, detection in enumerate(result.detections):
        x, y, width, height = detection.bbox
        color = (0, 0, 255) if index == 0 else (0, 255, 0)
        thickness = 3 if index == 0 else 2
        cv2.rectangle(
            visualization,
            (x, y),
            (x + width, y + height),
            color,
            thickness,
        )
    return visualization
