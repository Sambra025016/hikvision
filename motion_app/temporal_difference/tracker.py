# -*- coding: utf-8 -*-
"""
卡尔曼滤波跟踪模块（课题B 第二阶段，需求3/4）

在 MotionDetector 检测结果之上，对"最显著目标"做单目标跟踪：
    状态量 x = [cx, cy, vx, vy]T   （目标框中心坐标 + 速度，像素 / 帧）
    观测量 z = [cx, cy]T           （检测框中心）
    运动模型：匀速模型（CV），离散白噪声加速度驱动

关键设计：
    1. 数据关联：跟踪建立后，不再简单取"面积最大"的检测框，而是在预测位置
       周围设置关联门限（gate），取门限内尺寸一致且距离最近的检测框
       作为观测，避免多目标场景下跟踪目标跳变或被粘连块夺走；
    2. 丢失续航：某帧未关联到检测时仅用预测值续航（coast），且续航期间
       关联门限加倍以便于重新捕获目标；连续丢失超过 max_lost 帧
       （由调用方按 秒数*fps 换算）才判定"目标彻底丢失"，
       保证短暂遮挡/停顿下跟踪不中断，满足"至少跟踪10秒"；
    3. 轨迹预测：predict_future() 由当前滤波状态外推未来若干帧轨迹，
       供画面绘制与后续云台控制闭环使用；
    4. 误差记录：每帧记录"预测位置-实际检测位置"（预测误差）与
       "滤波位置-实际检测位置"（滤波误差），用于绘制跟踪误差曲线；
    5. 静止杂散识别：若观测中心在时间窗内几乎不动（锁到鬼影/静止杂散），
       is_static() 返回 True，由调用方丢弃该跟踪段并黑名单该区域。
"""

import cv2
import numpy as np


class KalmanTracker:
    """匀速模型卡尔曼单目标跟踪器"""

    def __init__(self, box, track_id=0,
                 max_lost=75,        # 允许连续丢失帧数（超过则目标彻底丢失；建议按 秒数*fps 传入）
                 gate_scale=0.6,     # 关联门限 = max(min_gate, gate_scale * 目标长边)
                 min_gate=50.0,      # 关联门限下限（像素）
                 lost_gate_mult=1.3, # 丢失续航期间门限放大倍数(过大会误关联邻人)
                 size_ratio=(0.5, 2.2),  # 关联候选与当前目标框的面积比允许范围
                 accel_std=0.5,      # 过程噪声：加速度标准差（像素/帧^2）
                 meas_std=3.0):      # 观测噪声：检测中心抖动标准差（像素）
        x, y, w, h = [float(v) for v in box]
        cx, cy = x + w / 2.0, y + h / 2.0

        self.kf = cv2.KalmanFilter(4, 2, 0)
        dt = 1.0  # 以"帧"为时间单位
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        # 离散白噪声加速度模型的过程噪声协方差
        q = accel_std ** 2
        self.kf.processNoiseCov = np.array(
            [[0.25, 0, 0.5, 0],
             [0, 0.25, 0, 0.5],
             [0.5, 0, 1, 0],
             [0, 0.5, 0, 1]], np.float32) * q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_std ** 2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000.0
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

        self.box = (x, y, w, h)          # 最近一次目标框（丢失时由预测平移）
        self.track_id = track_id
        self.max_lost = max_lost
        self.gate_scale = gate_scale
        self.min_gate = min_gate
        self.lost_gate_mult = lost_gate_mult
        self.size_ratio = size_ratio
        self.age = 0                     # 跟踪建立以来总帧数
        self.hits = 0                    # 关联到观测的帧数
        self.lost = 0                    # 当前连续丢失帧数
        self.active = True               # 是否仍在跟踪
        self.death_reason = 'lost'       # 结束原因: 'lost' 彻底丢失 / 'static' 静止杂散
        self._meas_hist = []             # 近期观测中心 [(age, cx, cy)]
        self.pred = (cx, cy)             # 本帧预测位置（修正前）
        self.filt = (cx, cy)             # 本帧滤波位置（修正后）

    # ---- 关联门限：随目标尺寸自适应，兼顾远近目标 ----
    def _gate(self):
        return max(self.min_gate, self.gate_scale * max(self.box[2], self.box[3]))

    def associate(self, boxes):
        """在预测位置门限内选取尺寸一致且中心最近的检测框；无匹配返回 None"""
        if not boxes:
            return None
        gate = self._gate()
        if self.lost > 0:
            gate *= self.lost_gate_mult  # 续航期间适度放大门限(过大会误关联邻人)
        own_area = self.box[2] * self.box[3]
        px, py = self.pred
        best, best_d = None, float('inf')
        for b in boxes:
            # 尺寸一致性: 拒绝面积相差过大的粘连块/碎片, 防止跟踪被夺走
            ba = b[2] * b[3]
            if not (self.size_ratio[0] * own_area <= ba <= self.size_ratio[1] * own_area):
                continue
            cx = b[0] + b[2] / 2.0
            cy = b[1] + b[3] / 2.0
            d = np.hypot(cx - px, cy - py)
            if d < best_d:
                best, best_d = b, d
        return best if best_d <= gate else None

    def update(self, boxes):
        """
        执行一帧的 预测 -> 关联 -> 修正。

        参数:
            boxes: 本帧全部运动目标检测框 [(x, y, w, h), ...]

        返回:
            meas : 关联到的检测框，未关联到为 None
            pred : 本帧预测位置 (cx, cy)（修正前）
            filt : 本帧滤波位置 (cx, cy)（修正后）
        """
        self.age += 1
        pred_state = self.kf.predict()
        self.pred = (float(pred_state[0]), float(pred_state[1]))

        meas = self.associate(boxes)
        if meas is not None:
            cx = meas[0] + meas[2] / 2.0
            cy = meas[1] + meas[3] / 2.0
            post = self.kf.correct(
                np.array([[np.float32(cx)], [np.float32(cy)]], np.float32))
            self.filt = (float(post[0]), float(post[1]))
            self.box = (float(meas[0]), float(meas[1]),
                        float(meas[2]), float(meas[3]))
            self.hits += 1
            self.lost = 0
            # 记录近期观测中心, 供静止杂散识别
            self._meas_hist.append((self.age, cx, cy))
            while self._meas_hist and self.age - self._meas_hist[0][0] > 50:
                self._meas_hist.pop(0)
        else:
            # 无观测：滤波值即预测值，目标框按预测中心平移（续航显示）
            self.filt = self.pred
            bw, bh = self.box[2], self.box[3]
            self.box = (self.filt[0] - bw / 2.0, self.filt[1] - bh / 2.0, bw, bh)
            # 续航期间衰减速度估计：目标停顿时预测不致漂走，
            # 恢复运动后检测仍能落入门限内被重新捕获
            self.kf.statePost[2] *= 0.9
            self.kf.statePost[3] *= 0.9
            self.lost += 1
            if self.lost > self.max_lost:
                self.active = False
                self.death_reason = 'lost'
        return meas, self.pred, self.filt

    def is_static(self, min_hits=10, eps=3.0):
        """
        静止杂散识别: 近期观测中心几乎不动(最大偏离均值 < eps 像素)
        且观测新鲜时返回 True —— 说明锁到了鬼影/静止杂散而非运动目标。
        """
        if len(self._meas_hist) < min_hits:
            return False
        if self.age - self._meas_hist[-1][0] > 10:
            return False  # 观测不新鲜(正在续航), 不作静止判定
        pts = np.array([[p[1], p[2]] for p in self._meas_hist], np.float32)
        center = pts.mean(axis=0)
        dev = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
        return float(dev.max()) < eps

    def predict_future(self, steps=20):
        """由当前滤波状态按匀速模型外推未来 steps 帧的轨迹点列表"""
        x, y = self.filt
        vx = float(self.kf.statePost[2])
        vy = float(self.kf.statePost[3])
        return [(x + vx * k, y + vy * k) for k in range(1, steps + 1)]

    @property
    def velocity(self):
        """当前速度估计（像素/帧）"""
        return float(self.kf.statePost[2]), float(self.kf.statePost[3])
