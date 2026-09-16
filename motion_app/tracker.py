"""基于匀速模型的单目标卡尔曼跟踪器。"""

from typing import Optional, Sequence

import cv2
import numpy as np

from motion_app.models import (
    BBox,
    Detection,
    TargetObservation,
    TrackState,
)
from motion_app.selector import TargetSelector


class KalmanTargetTracker:
    """完成预测、候选关联、测量修正和短时丢失续航。"""

    def __init__(self, selector: TargetSelector, max_misses=12,
                 max_prediction_seconds=0.8, prediction_horizon=0.18,
                 process_noise=3.0, measurement_noise=6.0,
                 initial_covariance=100.0, bbox_smoothing=0.25):
        self.selector = selector
        self.max_misses = int(max_misses)
        self.max_prediction_seconds = float(max_prediction_seconds)
        self.prediction_horizon = float(prediction_horizon)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_covariance = float(initial_covariance)
        self.bbox_smoothing = float(bbox_smoothing)
        self._next_track_id = 1
        self.reset(keep_id_counter=True)

    def reset(self, keep_id_counter=True):
        """清空当前轨迹；默认不复用已经分配过的 track_id。"""
        if not keep_id_counter:
            self._next_track_id = 1
        self._kf = None
        self._track_id = None
        self._bbox_size = None
        self._last_time = None
        self._start_time = None
        self._last_measurement_time = None
        self._miss_count = 0
        self._last_confidence = 0.0

    @property
    def active(self):
        return self._kf is not None and self._track_id is not None

    @property
    def current_bbox(self) -> Optional[BBox]:
        if not self.active or self._bbox_size is None:
            return None
        center = self._kf.statePost[:2].reshape(2)
        return self._bbox_from_center(center, self._bbox_size)

    @staticmethod
    def _bbox_from_center(center, size):
        width, height = size
        return (
            int(round(float(center[0]) - width / 2.0)),
            int(round(float(center[1]) - height / 2.0)),
            max(1, int(round(width))),
            max(1, int(round(height))),
        )

    def _initialize(self, detection, confidence, capture_time):
        self._kf = cv2.KalmanFilter(4, 2, 0)
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (
            self.measurement_noise ** 2
        )
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * (
            self.initial_covariance
        )
        self._kf.statePost = np.array(
            [[detection.center[0]], [detection.center[1]], [0.0], [0.0]],
            dtype=np.float32,
        )
        self._set_time_model(1.0 / 15.0)
        self._track_id = self._next_track_id
        self._next_track_id += 1
        self._bbox_size = (
            float(detection.bbox[2]), float(detection.bbox[3])
        )
        self._last_time = capture_time
        self._start_time = capture_time
        self._last_measurement_time = capture_time
        self._miss_count = 0
        self._last_confidence = confidence

    def _set_time_model(self, dt):
        dt = max(1.0 / 120.0, min(0.5, float(dt)))
        self._kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        q = self.process_noise ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        self._kf.processNoiseCov = np.array(
            [[dt4 / 4, 0, dt3 / 2, 0],
             [0, dt4 / 4, 0, dt3 / 2],
             [dt3 / 2, 0, dt2, 0],
             [0, dt3 / 2, 0, dt2]],
            dtype=np.float32,
        ) * q

    @staticmethod
    def _empty_state(frame_id, capture_time):
        return TrackState(
            frame_id=frame_id,
            capture_time=capture_time,
            track_id=None,
            bbox=None,
            measured_center=None,
            filtered_center=None,
            predicted_center=None,
            velocity=(0.0, 0.0),
            prediction_only=False,
            prediction_age=0.0,
            miss_count=0,
            active=False,
            confidence=0.0,
        )

    def update(self, frame_id: int, capture_time: float,
               detections: Sequence[Detection]):
        """处理一帧候选，返回 TargetObservation 和 TrackState。"""
        if not self.active:
            selected, confidence = self.selector.select_initial(detections)
            if selected is None:
                return None, self._empty_state(frame_id, capture_time)
            self._initialize(selected, confidence, capture_time)
            observation = TargetObservation(
                track_id=self._track_id,
                bbox=selected.bbox,
                center=selected.center,
                source="detection",
                confidence=confidence,
                visible=True,
                age=0.0,
                miss_count=0,
                source_frame_id=frame_id,
            )
            state = TrackState(
                frame_id=frame_id,
                capture_time=capture_time,
                track_id=self._track_id,
                bbox=selected.bbox,
                measured_center=selected.center,
                filtered_center=selected.center,
                predicted_center=selected.center,
                velocity=(0.0, 0.0),
                prediction_only=False,
                prediction_age=0.0,
                miss_count=0,
                active=True,
                confidence=confidence,
            )
            return observation, state

        dt = capture_time - self._last_time
        self._set_time_model(dt)
        predicted_state = self._kf.predict()
        predicted_now = (
            float(predicted_state[0]), float(predicted_state[1])
        )
        previous_bbox = self._bbox_from_center(
            predicted_now, self._bbox_size
        )
        gate_multiplier = 1.0 + min(2.0, self._miss_count * 0.15)
        selected, confidence = self.selector.associate(
            detections,
            predicted_now,
            previous_bbox,
            gate_multiplier=gate_multiplier,
        )

        measured_center = None
        observation = None
        prediction_only = selected is None
        if selected is not None:
            measurement = np.array(
                [[selected.center[0]], [selected.center[1]]],
                dtype=np.float32,
            )
            corrected = self._kf.correct(measurement)
            filtered = (float(corrected[0]), float(corrected[1]))
            measured_center = selected.center
            alpha = self.bbox_smoothing
            self._bbox_size = (
                (1.0 - alpha) * self._bbox_size[0] +
                alpha * selected.bbox[2],
                (1.0 - alpha) * self._bbox_size[1] +
                alpha * selected.bbox[3],
            )
            self._miss_count = 0
            self._last_measurement_time = capture_time
            self._last_confidence = confidence
            observation = TargetObservation(
                track_id=self._track_id,
                bbox=selected.bbox,
                center=selected.center,
                source="detection",
                confidence=confidence,
                visible=True,
                age=max(0.0, capture_time - self._start_time),
                miss_count=0,
                source_frame_id=frame_id,
            )
        else:
            filtered = predicted_now
            # 没有观测时仍要把预测状态提交为下一帧的先验。
            self._kf.statePost = predicted_state.copy()
            self._miss_count += 1

        self._last_time = capture_time
        prediction_age = max(
            0.0, capture_time - self._last_measurement_time
        )
        velocity = (
            float(self._kf.statePost[2]), float(self._kf.statePost[3])
        )
        predicted_future = (
            filtered[0] + velocity[0] * self.prediction_horizon,
            filtered[1] + velocity[1] * self.prediction_horizon,
        )
        bbox = self._bbox_from_center(filtered, self._bbox_size)

        if (self._miss_count > self.max_misses or
                prediction_age > self.max_prediction_seconds):
            lost_track_id = self._track_id
            lost_miss_count = self._miss_count
            self.reset(keep_id_counter=True)
            lost_state = self._empty_state(frame_id, capture_time)
            lost_state = TrackState(
                frame_id=lost_state.frame_id,
                capture_time=lost_state.capture_time,
                track_id=lost_track_id,
                bbox=None,
                measured_center=None,
                filtered_center=None,
                predicted_center=None,
                velocity=(0.0, 0.0),
                prediction_only=True,
                prediction_age=prediction_age,
                miss_count=lost_miss_count,
                active=False,
                confidence=0.0,
            )
            return None, lost_state

        if observation is None:
            decayed_confidence = self._last_confidence * max(
                0.0,
                1.0 - prediction_age / self.max_prediction_seconds,
            )
            observation = TargetObservation(
                track_id=self._track_id,
                bbox=bbox,
                center=filtered,
                source="kalman_prediction",
                confidence=decayed_confidence,
                visible=False,
                age=max(0.0, capture_time - self._start_time),
                miss_count=self._miss_count,
                source_frame_id=frame_id,
            )
            confidence = decayed_confidence

        state = TrackState(
            frame_id=frame_id,
            capture_time=capture_time,
            track_id=self._track_id,
            bbox=bbox,
            measured_center=measured_center,
            filtered_center=filtered,
            predicted_center=predicted_future,
            velocity=velocity,
            prediction_only=prediction_only,
            prediction_age=prediction_age,
            miss_count=self._miss_count,
            active=True,
            confidence=confidence,
        )
        return observation, state
