# -*- coding: utf-8 -*-
"""
自动抓拍模块（课题B 第三阶段，需求5）

抓拍逻辑：
    1. 中心判定：跟踪目标框中心与画面中心的偏移量在阈值比例内
       （center_ratio，相对画面宽/高）时，认为"人在画面中间"；
    2. 质量判别：对目标 ROI 计算 Laplacian 方差作为清晰度分数，
       分数 >= min_sharpness 才抓拍当前帧；
       分数不达标则顺延到后续帧继续评判（目标停留在中心区域内期间）；
    3. 单张约束：每个跟踪段（track_id）最多抓拍 1 张；
    4. 输出：抓拍图片保存到 capture/ 目录（文件名含视频名前缀，
       避免多视频互相覆盖），并汇总 CSV（含清晰度分数与中心偏移）。
"""

import csv
import os

import cv2


class AutoCapturer:
    """中心触发 + 清晰度判别的自动抓拍器"""

    def __init__(self, save_dir,
                 center_ratio=0.25,    # 中心区域半宽/半高占画面的比例
                 min_sharpness=500.0,  # Laplacian 方差清晰度阈值
                 prefix=''):           # 输出文件名前缀（一般取视频名）
        self.save_dir = save_dir
        self.center_ratio = center_ratio
        self.min_sharpness = min_sharpness
        self.prefix = prefix
        self.records = []            # 抓拍记录列表
        self.last_event = None       # 最近一次抓拍事件（供画面标注，由调用方清除）
        self._track_id = None
        self._captured = False       # 当前跟踪段是否已抓拍
        self._wait = 0               # 进入中心区域后因清晰度不足顺延的帧数

    # ---- 清晰度: 目标 ROI 的 Laplacian 方差, 运动模糊/失焦时显著下降 ----
    def sharpness(self, frame, box):
        x, y, w, h = [int(round(v)) for v in box]
        H, W = frame.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, W), min(y + h, H)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0
        roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(roi, cv2.CV_64F).var())

    def center_offset(self, box, frame_shape):
        """返回 (是否在中心区域, dx, dy): 目标中心相对画面中心的偏移"""
        H, W = frame_shape[:2]
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        dx, dy = cx - W / 2.0, cy - H / 2.0
        in_zone = (abs(dx) <= self.center_ratio * W
                   and abs(dy) <= self.center_ratio * H)
        return in_zone, dx, dy

    def update(self, frame, box, track_id, frame_idx, time_s):
        """
        每帧评判一次是否抓拍。

        返回: 抓拍成功时返回记录 dict（含保存路径与清晰度分数），否则 None
        """
        if track_id != self._track_id:      # 新跟踪段: 重置单张约束
            self._track_id = track_id
            self._captured = False
            self._wait = 0
        if self._captured:
            return None

        in_zone, dx, dy = self.center_offset(box, frame.shape)
        if not in_zone:
            self._wait = 0
            return None

        score = self.sharpness(frame, box)
        if score < self.min_sharpness:
            # 清晰度不达标: 顺延到后续帧继续评判
            self._wait += 1
            return None

        # ---- 抓拍当前帧（保存无标注原图） ----
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(
            self.save_dir,
            '{}track{}_capture.png'.format(self.prefix, track_id))
        cv2.imwrite(path, frame)
        self._captured = True
        rec = {'frame': frame_idx, 'time_s': time_s, 'track_id': track_id,
               'sharpness': round(score, 2), 'wait_frames': self._wait,
               'center_dx': round(dx, 1), 'center_dy': round(dy, 1),
               'file': os.path.basename(path)}
        self.records.append(rec)
        self.last_event = rec
        print('[抓拍] 帧{} track#{} 清晰度{:.1f} -> {}'.format(
            frame_idx, track_id, score, path))
        return rec

    def save_csv(self, csv_path):
        """保存抓拍记录（含清晰度分数）到 CSV"""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'frame', 'time_s', 'track_id', 'sharpness', 'wait_frames',
                'center_dx', 'center_dy', 'file'])
            w.writeheader()
            for r in self.records:
                w.writerow(r)
        print('抓拍记录已保存: {} (共{}张)'.format(csv_path, len(self.records)))
