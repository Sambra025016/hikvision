# -*- coding: utf-8 -*-
"""
卡尔曼跟踪主程序（课题B 第二阶段，需求3/4）

流程:
    MotionDetector 检测运动目标 -> KalmanTracker 对最显著目标建立跟踪
    -> 每帧记录 预测位置/滤波位置/检测位置 与跟踪误差
    -> 画面绘制: 跟踪框、实测轨迹、滤波轨迹、未来轨迹预测
    -> AutoCapturer 抓拍判定(需求5): 目标中心进入画面中心区域且
       ROI 清晰度(Laplacian方差)达标时自动抓拍1张, 不达标顺延后续帧
    -> 输出: 连续跟踪视频 output/track_<name>.mp4
             误差数据   output/track_<name>_error.csv
             误差曲线   output/track_<name>_error.png
             抓拍图片   capture/<name>_track<X>_capture.png + <name>_captures.csv
    -> 统计连续跟踪时长，检验是否满足"稳定跟踪10秒"（需求4）；
       跟踪段仅在"目标彻底丢失"(连续 max_lost_sec 秒无关联)或视频结束
       时才结束，否则同一目标持续跟踪（至少10秒）

用法:
    python run_tracking.py                          # 处理 video/ 下所有视频
    python run_tracking.py --video video/003.avi    # 只处理指定视频
    python run_tracking.py --no-show                # 不显示窗口（headless）
    python run_tracking.py --no-capture             # 关闭自动抓拍

按键: q / ESC 退出当前视频
"""

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motion_app.models import FramePacket
from motion_app.preprocess import FramePreprocessor
from motion_app.temporal_difference.capturer import AutoCapturer
from motion_app.temporal_difference.detector import (
    IdentityFramePairBuilder,
    MotionDetector,
)
from motion_app.temporal_difference.tracker import KalmanTracker

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'capture')
FUTURE_STEPS = 20          # 画面外推的未来轨迹帧数
REQUIRED_TRACK_SEC = 10.0  # 需求4: 稳定跟踪时长要求

# 绘制颜色 (BGR)
C_MEAS = (0, 255, 0)       # 检测/实测轨迹: 绿
C_FILT = (255, 0, 0)       # 滤波轨迹: 蓝
C_PRED = (0, 255, 255)     # 预测位置/未来轨迹: 黄
C_TRACK = (0, 0, 255)      # 跟踪框: 红
C_COAST = (0, 128, 255)    # 丢失续航框: 橙


def _polyline(vis, pts, color, thickness=2):
    if len(pts) < 2:
        return
    arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in pts],
                   np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [arr], False, color, thickness)


def pick_salient(boxes, blacklist, n):
    """在多目标中选取最显著目标(面积最大), 跳过黑名单区域(静止杂散)"""
    cand = []
    for b in boxes:
        cx = b[0] + b[2] / 2.0
        cy = b[1] + b[3] / 2.0
        if any(n < exp and np.hypot(cx - bx, cy - by) < r
               for bx, by, r, exp in blacklist):
            continue
        cand.append(b)
    return max(cand, key=lambda b: b[2] * b[3]) if cand else None


def draw_tracking(vis, tracker, meas, hist, future):
    """在画面副本上绘制跟踪结果(不修改原帧, 保证抓拍图为无标注原图)"""
    vis = vis.copy()
    # 历史轨迹: 实测中心(绿) + 滤波中心(蓝)
    _polyline(vis, hist['meas'], C_MEAS, 1)
    _polyline(vis, hist['filt'], C_FILT, 2)
    # 未来轨迹预测(黄): 连线 + 点
    _polyline(vis, [tracker.filt] + list(future), C_PRED, 1)
    for p in future[::2]:
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 2, C_PRED, -1)

    # 跟踪框: 关联到检测为红实线, 丢失续航为橙虚线效果(细线)
    x, y, w, h = [int(round(v)) for v in tracker.box]
    color = C_TRACK if meas is not None else C_COAST
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3 if meas else 2)

    # 本帧预测位置(黄叉) 与 滤波位置(蓝点)
    px, py = int(round(tracker.pred[0])), int(round(tracker.pred[1]))
    cv2.drawMarker(vis, (px, py), C_PRED, cv2.MARKER_CROSS, 14, 2)
    fx, fy = int(round(tracker.filt[0])), int(round(tracker.filt[1]))
    cv2.circle(vis, (fx, fy), 4, C_FILT, -1)
    if meas is not None:
        mx = int(round(meas[0] + meas[2] / 2.0))
        my = int(round(meas[1] + meas[3] / 2.0))
        cv2.circle(vis, (mx, my), 4, C_MEAS, -1)

    # HUD 信息
    vx, vy = tracker.velocity
    hud = ['track#{}  age:{}s  lost:{}'.format(
               tracker.track_id, tracker.age, tracker.lost),
           'v=({:.1f},{:.1f}) px/f'.format(vx, vy)]
    for i, s in enumerate(hud):
        cv2.putText(vis, s, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


def save_error_curve(rows, segments, fps, png_path):
    """用 matplotlib 绘制跟踪误差曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t, ep, ef = [], [], []
    for r in rows:
        if r['err_pred_px'] == '':
            continue
        t.append(r['time_s'])
        ep.append(r['err_pred_px'])
        ef.append(r['err_filt_px'])
    if not t:
        print('[警告] 无误差数据, 跳过误差曲线: {}'.format(png_path))
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, ep, '-', color='tab:orange', lw=1.2,
            label='prediction error (pred - meas)')
    ax.plot(t, ef, '-', color='tab:blue', lw=1.2,
            label='filter error (filt - meas)')
    for seg in segments:
        ax.axvline((seg['start'] - 1) / fps, color='gray',
                   ls='--', lw=0.8, alpha=0.7)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('error (pixel)')
    ax.set_title('Tracking error curve (gray dash = track re-init)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print('误差曲线已保存: {}'.format(png_path))


def process_video_tracking(video_path, show=True, future_steps=FUTURE_STEPS,
                           capture=True, sharpness=500.0, center_ratio=0.25,
                           max_lost_sec=8.0):
    """对单个视频执行 检测+卡尔曼跟踪, 输出跟踪视频/误差数据/统计"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('[错误] 视频无法打开: {}'.format(video_path))
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    name = os.path.splitext(os.path.basename(video_path))[0]
    print('跟踪视频: {}  size: {}  fps: {:.1f}'.format(video_path, size, fps))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_out = os.path.join(OUTPUT_DIR, 'track_{}.mp4'.format(name))
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, size)

    preprocessor = FramePreprocessor(target_size=size)
    pair_builder = IdentityFramePairBuilder()
    detector = MotionDetector()
    capturer = AutoCapturer(CAPTURE_DIR, center_ratio=center_ratio,
                            min_sharpness=sharpness,
                            prefix=name + '_') if capture else None
    tracker, next_id = None, 0
    segments, rows = [], []
    blacklist = []  # 静止杂散区域 [(cx, cy, radius, expire_frame)]
    hist = {'meas': [], 'filt': []}
    n = 0

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        n += 1
        packet = FramePacket(
            frame_id=n,
            capture_time=time.monotonic(),
            bgr_frame=frame,
            source_width=size[0],
            source_height=size[1],
            stream_ok=True,
        )
        processed = preprocessor.process(packet)
        if processed is None:
            pair_builder.reset()
            continue
        frame = processed.display_frame
        difference_input = pair_builder.build(processed)
        detection_result = detector.detect(difference_input)
        boxes = [item.bbox for item in detection_result.detections]
        salient = boxes[0] if boxes else None
        meas = None
        blacklist = [e for e in blacklist if n < e[3]]
        salient = pick_salient(boxes, blacklist, n)

        created_now = False
        if tracker is None or not tracker.active:
            # 上一段跟踪已结束(或尚未建立): 用显著目标重新初始化
            if tracker is not None:
                segments[-1]['reason'] = tracker.death_reason
            if salient is not None:
                tracker = KalmanTracker(
                    salient, track_id=next_id,
                    max_lost=max(1, int(round(max_lost_sec * fps))))
                next_id += 1
                created_now = True
                segments.append({'id': tracker.track_id,
                                 'start': n, 'end': n, 'hits': 1})
                c = (salient[0] + salient[2] / 2.0, salient[1] + salient[3] / 2.0)
                hist = {'meas': [c], 'filt': [c]}
                rows.append({'frame': n, 'time_s': round((n - 1) / fps, 4),
                             'track_id': tracker.track_id,
                             'meas_x': round(c[0], 1), 'meas_y': round(c[1], 1),
                             'pred_x': round(c[0], 1), 'pred_y': round(c[1], 1),
                             'filt_x': round(c[0], 1), 'filt_y': round(c[1], 1),
                             'err_pred_px': '', 'err_filt_px': ''})
            else:
                tracker = None
        if tracker is not None:
            if not created_now:  # 非本帧新建: 走 预测-关联-修正
                meas, pred, filt = tracker.update(boxes)
                segments[-1]['end'] = n
                if tracker.is_static():
                    # 锁到静止杂散(鬼影等): 丢弃本段并黑名单该区域
                    tracker.active = False
                    tracker.death_reason = 'static'
                    bx, by, bw, bh = tracker.box
                    blacklist.append((bx + bw / 2.0, by + bh / 2.0,
                                      max(bw, bh), n + int(10 * fps)))
                if meas is not None:
                    segments[-1]['hits'] += 1
                    mc = (meas[0] + meas[2] / 2.0, meas[1] + meas[3] / 2.0)
                    hist['meas'].append(mc)
                    err_p = float(np.hypot(pred[0] - mc[0], pred[1] - mc[1]))
                    err_f = float(np.hypot(filt[0] - mc[0], filt[1] - mc[1]))
                else:
                    mc, err_p, err_f = None, None, None
                hist['filt'].append(filt)
                rows.append({
                    'frame': n, 'time_s': round((n - 1) / fps, 4),
                    'track_id': tracker.track_id,
                    'meas_x': round(mc[0], 1) if mc else '',
                    'meas_y': round(mc[1], 1) if mc else '',
                    'pred_x': round(pred[0], 1), 'pred_y': round(pred[1], 1),
                    'filt_x': round(filt[0], 1), 'filt_y': round(filt[1], 1),
                    'err_pred_px': round(err_p, 2) if err_p is not None else '',
                    'err_filt_px': round(err_f, 2) if err_f is not None else ''})
            else:
                meas = salient  # 本帧新建跟踪
            future = tracker.predict_future(future_steps)
            vis = draw_tracking(frame, tracker, meas, hist, future)
        else:
            vis = frame.copy()
            cv2.putText(vis, 'waiting for target...', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---- 抓拍判定(需求5): 中心区域 + 清晰度达标才拍, 否则顺延 ----
        if capturer is not None and tracker is not None and meas is not None:
            event = capturer.update(frame, tracker.box, tracker.track_id,
                                    n, (n - 1) / fps)
            if event is not None:
                H, W = vis.shape[:2]
                cv2.rectangle(vis, (2, 2), (W - 3, H - 3), (255, 255, 255), 4)
                cv2.putText(vis, 'CAPTURED! sharpness={:.1f}'.format(
                                event['sharpness']), (10, H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                capturer.last_event = None

        writer.write(vis)
        if show:
            cv2.imshow('kalman tracking', vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    writer.release()
    if segments:
        segments[-1].setdefault(
            'reason', tracker.death_reason if not tracker.active else 'eof')
    print('跟踪视频已保存: {}'.format(video_out))

    # ---- 误差数据 CSV ----
    csv_path = os.path.join(OUTPUT_DIR, 'track_{}_error.csv'.format(name))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'frame', 'time_s', 'track_id', 'meas_x', 'meas_y',
            'pred_x', 'pred_y', 'filt_x', 'filt_y',
            'err_pred_px', 'err_filt_px'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('误差数据已保存: {}'.format(csv_path))

    # ---- 连续跟踪统计 ----
    print('---- 连续跟踪统计 (需求4: >= {:.0f}s) ----'.format(REQUIRED_TRACK_SEC))
    best = 0.0
    for seg in segments:
        dur = (seg['end'] - seg['start'] + 1) / fps
        best = max(best, dur)
        reason = {'lost': '目标彻底丢失',
                  'static': '静止杂散丢弃'}.get(seg.get('reason'), '视频结束')
        print('  track#{}: 帧 {}~{}  连续跟踪 {:.2f}s  观测率 {:.0%}  结束原因: {}'.format(
            seg['id'], seg['start'], seg['end'], dur,
            seg['hits'] / (seg['end'] - seg['start'] + 1), reason))
    if segments:
        print('最长连续跟踪: {:.2f}s  {}'.format(
            best, '满足' if best >= REQUIRED_TRACK_SEC else '不满足(视频过短或跟踪中断)'))

    # ---- 误差曲线 ----
    save_error_curve(rows, segments, fps,
                     os.path.join(OUTPUT_DIR, 'track_{}_error.png'.format(name)))

    # ---- 抓拍汇总 ----
    if capturer is not None:
        capturer.save_csv(
            os.path.join(CAPTURE_DIR, '{}_captures.csv'.format(name)))
    return best


def main():
    parser = argparse.ArgumentParser(description='卡尔曼滤波运动目标跟踪')
    parser.add_argument('--video', type=str, default=None, help='指定视频路径')
    parser.add_argument('--no-show', action='store_true', help='不显示窗口')
    parser.add_argument('--future', type=int, default=FUTURE_STEPS,
                        help='未来轨迹外推帧数')
    parser.add_argument('--no-capture', action='store_true', help='关闭自动抓拍')
    parser.add_argument('--sharpness', type=float, default=500.0,
                        help='抓拍清晰度阈值(Laplacian方差)')
    parser.add_argument('--center-ratio', type=float, default=0.25,
                        help='中心判定区域半宽/半高占画面比例')
    parser.add_argument('--max-lost-sec', type=float, default=8.0,
                        help='连续丢失超过该秒数才判定目标彻底丢失')
    args = parser.parse_args()

    if args.video:
        video_list = [args.video]
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video')
        video_list = sorted(glob.glob(os.path.join(base, '*.avi'))) + \
                     sorted(glob.glob(os.path.join(base, '*.mp4')))
    if not video_list:
        print('[错误] 未找到待处理视频')
        return

    for path in video_list:
        process_video_tracking(path, show=not args.no_show,
                               future_steps=args.future,
                               capture=not args.no_capture,
                               sharpness=args.sharpness,
                               center_ratio=args.center_ratio,
                               max_lost_sec=args.max_lost_sec)

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # 无 GUI 环境（headless）下忽略


if __name__ == '__main__':
    main()
