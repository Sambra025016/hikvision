"""运动候选的初始选择与跟踪期数据关联。"""

import math


class TargetSelector:
    """用位置、尺寸和运动强度选择最可能属于当前目标的候选。"""

    def __init__(self, association_gate=120.0, area_ratio_min=0.25,
                 area_ratio_max=4.0, distance_weight=0.65,
                 area_weight=0.20, motion_weight=0.15):
        self.association_gate = float(association_gate)
        self.area_ratio_min = float(area_ratio_min)
        self.area_ratio_max = float(area_ratio_max)
        self.distance_weight = float(distance_weight)
        self.area_weight = float(area_weight)
        self.motion_weight = float(motion_weight)

    @staticmethod
    def _area(bbox):
        return max(1.0, float(bbox[2] * bbox[3]))

    def select_initial(self, detections):
        """未建立跟踪时，以面积、填充率和运动强度选择显著目标。"""
        if not detections:
            return None, 0.0
        maximum_area = max(item.bbox_area for item in detections)
        scored = []
        for item in detections:
            area_score = item.bbox_area / float(maximum_area)
            score = (
                0.55 * area_score +
                0.20 * min(1.0, item.fill_ratio) +
                0.25 * min(1.0, item.motion_energy)
            )
            scored.append((score, item))
        score, selected = max(scored, key=lambda pair: pair[0])
        return selected, float(max(0.0, min(1.0, score)))

    def associate(self, detections, predicted_center, previous_bbox,
                  gate_multiplier=1.0):
        """在预测位置附近选择尺寸相符的候选，避免多人场景下跳目标。"""
        if not detections or predicted_center is None or previous_bbox is None:
            return None, 0.0
        gate = self.association_gate * max(1.0, float(gate_multiplier))
        previous_area = self._area(previous_bbox)
        best = None
        best_cost = float("inf")

        for item in detections:
            area_ratio = item.bbox_area / previous_area
            if not self.area_ratio_min <= area_ratio <= self.area_ratio_max:
                continue
            distance = math.hypot(
                item.center[0] - predicted_center[0],
                item.center[1] - predicted_center[1],
            )
            if distance > gate:
                continue
            distance_cost = distance / gate
            area_cost = min(1.0, abs(math.log(max(area_ratio, 1e-6))))
            motion_cost = 1.0 - min(1.0, item.motion_energy)
            cost = (
                self.distance_weight * distance_cost +
                self.area_weight * area_cost +
                self.motion_weight * motion_cost
            )
            if cost < best_cost:
                best = item
                best_cost = cost

        if best is None:
            return None, 0.0
        return best, float(max(0.0, min(1.0, 1.0 - best_cost)))

