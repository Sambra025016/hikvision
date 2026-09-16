"""模块之间共享的数据对象。

这些对象对应《课题B_运动目标检测与球机跟踪实现方案》3.2 节，避免各模块
使用含义不清的元组传递帧、检测框和状态。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]


@dataclass(frozen=True)
class FramePacket:
    """RTSP 采集模块发布的原始帧包。"""

    frame_id: int
    capture_time: float
    bgr_frame: Optional[Any]
    source_width: int
    source_height: int
    stream_ok: bool


@dataclass(frozen=True)
class ProcessedFrame:
    """预处理后供视觉算法使用的帧。"""

    frame_id: int
    capture_time: float
    display_frame: Any
    gray_frame: Any
    scale_x: float
    scale_y: float
    valid_roi: Any


@dataclass(frozen=True)
class CompensationOutput:
    """历史帧对齐到当前帧坐标系后的统一输出。

    两帧验证阶段使用恒等变换；后续全局运动补偿模块仍输出同一对象。
    """

    frame_id: int
    capture_time: float
    display_frame: Any
    current_gray: Any
    aligned_previous: Optional[Any]
    aligned_old: Optional[Any]
    valid_mask: Any
    transform_previous: Optional[Any]
    transform_old: Optional[Any]
    quality: Dict[str, Any] = field(default_factory=dict)
    compensation_ok: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Detection:
    """单个运动候选目标。"""

    bbox: BBox
    center: Point
    contour_area: float
    bbox_area: int
    fill_ratio: float
    motion_energy: float
    source_frame_id: int


@dataclass(frozen=True)
class DifferenceQuality:
    """当前帧差结果的质量与时序信息。"""

    threshold: int
    changed_pixel_ratio: float
    processing_time_ms: float
    previous_frame_id: Optional[int]
    frame_interval: Optional[float]


@dataclass(frozen=True)
class DetectionResult:
    """帧差检测器的完整输出。"""

    source_frame_id: int
    motion_mask: Any
    detections: Tuple[Detection, ...]
    quality: DifferenceQuality
    detection_ok: bool
    reason: str = ""


@dataclass(frozen=True)
class TargetObservation:
    """目标选择模块交给跟踪器的当前观测。"""

    track_id: int
    bbox: Optional[BBox]
    center: Optional[Point]
    source: str
    confidence: float
    visible: bool
    age: float
    miss_count: int
    source_frame_id: int


@dataclass(frozen=True)
class TrackState:
    """卡尔曼滤波更新后的目标状态。"""

    frame_id: int
    capture_time: float
    track_id: Optional[int]
    bbox: Optional[BBox]
    measured_center: Optional[Point]
    filtered_center: Optional[Point]
    predicted_center: Optional[Point]
    velocity: Point
    prediction_only: bool
    prediction_age: float
    miss_count: int
    active: bool
    confidence: float


@dataclass(frozen=True)
class ControlCommand:
    """目标位置到云台执行层之间的控制命令。"""

    command_time: float
    track_id: Optional[int]
    pan: int
    tilt: int
    zoom: int
    reason: str
    valid_until: float


@dataclass(frozen=True)
class SnapshotResult:
    """居中抓拍模块的单帧输出。"""

    captured: bool
    file_path: Optional[str]
    sharpness: float
    centered_frames: int
    reason: str


@dataclass(frozen=True)
class PipelineResult:
    """完整流水线处理一帧后产生的只读结果。"""

    packet: FramePacket
    processed: ProcessedFrame
    compensation: CompensationOutput
    detection: DetectionResult
    observation: Optional[TargetObservation]
    track: TrackState
    command: ControlCommand
    snapshot: SnapshotResult
    visualization: Any


@dataclass(frozen=True)
class StreamStatus:
    """RTSP 采集线程的只读状态快照。"""

    connected: bool
    frames_received: int
    dropped_by_overwrite: int
    read_failures: int
    reconnect_count: int
    measured_fps: float
    last_frame_time: Optional[float]
    last_error: str
