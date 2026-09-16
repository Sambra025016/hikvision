"""相邻帧全局运动补偿。

第一版支持恒等变换和 ORB + RANSAC 单应矩阵。目标框会从特征区域中排除，
最近的 PTZ 指令仅作为质量信息记录；真实变换始终由图像内容确认。
"""

from typing import Iterable, Optional

import cv2
import numpy as np

from motion_app.models import (
    BBox,
    CompensationOutput,
    ControlCommand,
    ProcessedFrame,
)


class GlobalMotionCompensator:
    """维护上一帧，并将它对齐到当前帧坐标系。"""

    def __init__(self, mode="identity", max_features=800, ratio_test=0.75,
                 min_matches=20, min_inlier_ratio=0.35,
                 max_reprojection_error=3.0, min_overlap_ratio=0.70,
                 edge_margin=12, exclude_padding=20,
                 allow_identity_fallback=True):
        self.mode = mode
        self.ratio_test = float(ratio_test)
        self.min_matches = int(min_matches)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_reprojection_error = float(max_reprojection_error)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.edge_margin = int(edge_margin)
        self.exclude_padding = int(exclude_padding)
        self.allow_identity_fallback = bool(allow_identity_fallback)
        self._orb = cv2.ORB_create(nfeatures=int(max_features))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._previous = None

    def reset(self):
        """断流、尺寸变化或重新开始实验时清空历史帧。"""
        self._previous = None

    @staticmethod
    def _waiting(current, reason="waiting_for_previous_frame"):
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
            quality={"mode": "none", "previous_frame_id": None},
            compensation_ok=False,
            reason=reason,
        )

    @staticmethod
    def _command_info(command: Optional[ControlCommand]):
        if command is None:
            return {"prior_pan": 0, "prior_tilt": 0, "prior_zoom": 0}
        return {
            "prior_pan": command.pan,
            "prior_tilt": command.tilt,
            "prior_zoom": command.zoom,
        }

    def _identity(self, previous, current, reason=""):
        height, width = current.gray_frame.shape
        valid_mask = cv2.bitwise_and(
            previous.valid_roi, current.valid_roi
        )
        return CompensationOutput(
            frame_id=current.frame_id,
            capture_time=current.capture_time,
            display_frame=current.display_frame,
            current_gray=current.gray_frame,
            aligned_previous=previous.gray_frame,
            aligned_old=None,
            valid_mask=valid_mask,
            transform_previous=np.eye(3, dtype=np.float32),
            transform_old=None,
            quality={
                "mode": "identity",
                "fallback_reason": reason,
                "previous_frame_id": previous.frame_id,
                "previous_capture_time": previous.capture_time,
                "overlap_ratio": cv2.countNonZero(valid_mask) /
                float(width * height),
            },
            compensation_ok=True,
            reason="",
        )

    def _feature_mask(self, processed, excluded_bboxes: Iterable[BBox]):
        mask = processed.valid_roi.copy()
        height, width = mask.shape
        margin = max(0, self.edge_margin)
        if margin:
            mask[:margin, :] = 0
            mask[-margin:, :] = 0
            mask[:, :margin] = 0
            mask[:, -margin:] = 0
        padding = max(0, self.exclude_padding)
        for x, y, box_width, box_height in excluded_bboxes:
            x1 = max(0, int(x) - padding)
            y1 = max(0, int(y) - padding)
            x2 = min(width, int(x + box_width) + padding)
            y2 = min(height, int(y + box_height) + padding)
            mask[y1:y2, x1:x2] = 0
        return mask

    def _orb_align(self, previous, current, excluded_bboxes, command):
        previous_mask = self._feature_mask(previous, excluded_bboxes)
        current_mask = self._feature_mask(current, excluded_bboxes)
        previous_points, previous_desc = self._orb.detectAndCompute(
            previous.gray_frame, previous_mask
        )
        current_points, current_desc = self._orb.detectAndCompute(
            current.gray_frame, current_mask
        )
        if previous_desc is None or current_desc is None:
            return None, "not_enough_features"

        pairs = self._matcher.knnMatch(previous_desc, current_desc, k=2)
        good = []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < self.ratio_test * pair[1].distance:
                good.append(pair[0])
        if len(good) < self.min_matches:
            return None, "not_enough_matches"

        source = np.float32([
            previous_points[item.queryIdx].pt for item in good
        ]).reshape(-1, 1, 2)
        destination = np.float32([
            current_points[item.trainIdx].pt for item in good
        ]).reshape(-1, 1, 2)
        transform, inlier_mask = cv2.findHomography(
            source, destination, cv2.RANSAC, self.max_reprojection_error
        )
        if transform is None or inlier_mask is None:
            return None, "homography_failed"

        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        inlier_ratio = inlier_count / float(len(good))
        if inlier_count < 4 or inlier_ratio < self.min_inlier_ratio:
            return None, "low_inlier_ratio"

        projected = cv2.perspectiveTransform(source[inliers], transform)
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - destination[inliers].reshape(-1, 2),
            axis=1,
        )
        median_error = float(np.median(errors)) if len(errors) else float("inf")
        if median_error > self.max_reprojection_error:
            return None, "high_reprojection_error"

        height, width = current.gray_frame.shape
        aligned = cv2.warpPerspective(
            previous.gray_frame, transform, (width, height)
        )
        previous_valid = cv2.warpPerspective(
            previous.valid_roi,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        valid_mask = cv2.bitwise_and(previous_valid, current.valid_roi)
        valid_mask = cv2.erode(
            valid_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        overlap_ratio = cv2.countNonZero(valid_mask) / float(width * height)
        if overlap_ratio < self.min_overlap_ratio:
            return None, "low_overlap_ratio"

        residual = cv2.absdiff(current.gray_frame, aligned)
        residual_mean = float(cv2.mean(residual, mask=valid_mask)[0])
        quality = {
            "mode": "orb",
            "previous_frame_id": previous.frame_id,
            "previous_capture_time": previous.capture_time,
            "match_count": len(good),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "median_reprojection_error": median_error,
            "overlap_ratio": overlap_ratio,
            "background_residual_mean": residual_mean,
        }
        quality.update(self._command_info(command))
        return CompensationOutput(
            frame_id=current.frame_id,
            capture_time=current.capture_time,
            display_frame=current.display_frame,
            current_gray=current.gray_frame,
            aligned_previous=aligned,
            aligned_old=None,
            valid_mask=valid_mask,
            transform_previous=transform,
            transform_old=None,
            quality=quality,
            compensation_ok=True,
            reason="",
        ), ""

    def process(self, current: ProcessedFrame, excluded_bboxes=(),
                command: Optional[ControlCommand] = None):
        """消费当前帧并输出上一帧到当前帧的对齐结果。"""
        previous = self._previous
        self._previous = current
        if previous is None:
            return self._waiting(current)
        if previous.gray_frame.shape != current.gray_frame.shape:
            return self._waiting(current, "frame_size_changed")
        if self.mode == "identity":
            result = self._identity(previous, current)
            result.quality.update(self._command_info(command))
            return result

        result, reason = self._orb_align(
            previous, current, excluded_bboxes, command
        )
        if result is not None:
            return result
        if self.allow_identity_fallback:
            fallback = self._identity(previous, current, reason)
            fallback.quality.update(self._command_info(command))
            return fallback
        unavailable = self._waiting(current, reason)
        unavailable.quality.update(self._command_info(command))
        return unavailable
