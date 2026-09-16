"""运动检测、目标跟踪和 PTZ 决策的主数据流。"""

import time

from motion_app.compensation import GlobalMotionCompensator
from motion_app.controller import DryRunPTZExecutor, PTZDecisionController
from motion_app.models import PipelineResult
from motion_app.preprocess import FramePreprocessor
from motion_app.selector import TargetSelector
from motion_app.snapshot import CenterSnapshotManager
from motion_app.temporal_difference.detector import MotionDetector
from motion_app.tracker import KalmanTargetTracker
from motion_app.visualization import draw_pipeline_result


class TrackingPipeline:
    """按固定顺序连接所有模块，并维护两条反馈信息。"""

    def __init__(self, config, capture_dir, executor=None):
        preprocess = config["preprocess"]
        self.preprocessor = FramePreprocessor(
            target_size=(preprocess["width"], preprocess["height"]),
            blur_size=tuple(preprocess["blur_size"]),
            roi=preprocess.get("roi"),
        )
        self.compensator = GlobalMotionCompensator(
            **config["compensation"]
        )
        detector_options = dict(config["detector"])
        detector_options["open_kernel"] = tuple(
            detector_options["open_kernel"]
        )
        detector_options["close_kernel"] = tuple(
            detector_options["close_kernel"]
        )
        self.detector = MotionDetector(**detector_options)
        self.selector = TargetSelector(**config["selector"])
        self.tracker = KalmanTargetTracker(
            self.selector, **config["tracker"]
        )
        self.controller = PTZDecisionController(**config["controller"])
        self.executor = executor or DryRunPTZExecutor()
        self.snapshot = CenterSnapshotManager(
            capture_dir, **config["snapshot"]
        )
        self._last_track = None
        self._last_command = None

    def reset(self, reason="stream_reset"):
        """断流或输入切换时清空所有依赖历史帧的状态并停止 PTZ。"""
        now = time.monotonic()
        track_id = self._last_track.track_id if self._last_track else None
        stop = self.controller.stop_command(now, track_id, reason)
        self.executor.execute(stop)
        self.compensator.reset()
        self.tracker.reset(keep_id_counter=True)
        self._last_track = None
        self._last_command = stop

    def process(self, packet):
        """处理一个 FramePacket；无效帧返回 None 并重置历史状态。"""
        if not packet.stream_ok or packet.bgr_frame is None:
            self.reset("stream_invalid")
            return None
        processed = self.preprocessor.process(packet)
        if processed is None:
            self.reset("preprocess_failed")
            return None

        excluded = []
        if self._last_track is not None and self._last_track.bbox is not None:
            excluded.append(self._last_track.bbox)
        compensation = self.compensator.process(
            processed,
            excluded_bboxes=excluded,
            command=self._last_command,
        )
        detection = self.detector.detect(compensation)
        detections = detection.detections if detection.detection_ok else tuple()
        observation, track = self.tracker.update(
            frame_id=packet.frame_id,
            capture_time=packet.capture_time,
            detections=detections,
        )
        height, width = processed.display_frame.shape[:2]
        command = self.controller.compute(
            track, frame_size=(width, height), now=time.monotonic()
        )
        self.executor.execute(command)
        snapshot = self.snapshot.update(
            packet, processed, observation, track, command
        )
        visualization = draw_pipeline_result(
            processed.display_frame,
            detection,
            track,
            command,
            compensation,
        )
        result = PipelineResult(
            packet=packet,
            processed=processed,
            compensation=compensation,
            detection=detection,
            observation=observation,
            track=track,
            command=command,
            snapshot=snapshot,
            visualization=visualization,
        )
        self._last_track = track
        self._last_command = command
        return result

    def close(self):
        self.reset("shutdown")
        self.executor.stop()

