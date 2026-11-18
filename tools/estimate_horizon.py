#!/usr/bin/env python3
"""Numeric-only provisional pitch from converging lower-image Hough segments.

Assumes visible orchard rows are parallel and level, zero roll, square pixels,
centred principal point and a supplied (assumed, not estimated) horizontal FOV.
This is a vanishing-row proxy, not a calibrated horizon detector.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def estimate(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    edges[:int(.35 * h)] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35,
                            minLineLength=max(30, w // 14), maxLineGap=20)
    if lines is None:
        return None
    segments = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dx == 0 or not .2 < abs(dy / dx) < 2.5:
            continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if my < .45 * h or not (mx < .48 * w and dy / dx < 0 or mx > .52 * w and dy / dx > 0):
            continue
        length = math.hypot(dx, dy)
        a, b = -dy / length, dx / length
        segments.append((a, b, -(a * x1 + b * y1), length, mx < w / 2))
    if len(segments) < 4:
        return None
    arr = np.asarray([s[:4] for s in segments])
    points = []
    for i, s in enumerate(segments):
        for t in segments[i + 1:]:
            if s[4] == t[4]:
                continue
            matrix = np.asarray([s[:2], t[:2]])
            if abs(np.linalg.det(matrix)) < .1:
                continue
            point = np.linalg.solve(matrix, -np.asarray([s[2], t[2]]))
            if .2 * w < point[0] < .8 * w and .05 * h < point[1] < .65 * h:
                points.append(point)
    if not points:
        return None
    # Deterministic consensus of opposing slopes, then least squares on inliers.
    points = np.asarray(points)
    residual = np.abs(points @ arr[:, :2].T + arr[:, 2])
    support = (residual < .025 * h) @ arr[:, 3]
    best = points[int(np.argmax(support))]
    inliers = np.abs(arr[:, :2] @ best + arr[:, 2]) < .025 * h
    if inliers.sum() < 4 or len({segments[i][4] for i in np.flatnonzero(inliers)}) < 2:
        return None
    fit, _, _, _ = np.linalg.lstsq(arr[inliers, :2], -arr[inliers, 2], rcond=None)
    if not .05 * h < fit[1] < .65 * h:
        return None
    return dict(horizon_row_fraction=float(fit[1] / h), vanishing_x_fraction=float(fit[0] / w),
                segment_count=len(segments), inlier_count=int(inliers.sum()),
                residual_px=float(np.median(np.abs(arr[inliers, :2] @ fit + arr[inliers, 2]))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('videos', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=24)
    p.add_argument('--horizontal-fov-deg', type=float, default=70.)
    args = p.parse_args()
    if args.samples < 2 or not 0 < args.horizontal_fov_deg < 180:
        p.error('samples >= 2 and FOV in (0,180) required')
    summaries, pitches = [], []
    for path in args.videos:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise OSError('cannot open ' + path)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        for index in np.linspace(0, max(0, total - 1), args.samples, dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if not ok:
                rows.append(dict(frame=int(index), accepted=False, reason='decode_failed'))
                continue
            h, w = image.shape[:2]
            scaled = cv2.resize(image, (960, round(h * 960 / w)))
            record = estimate(scaled)
            if record is None:
                rows.append(dict(frame=int(index), accepted=False, reason='no_consensus'))
                continue
            row = record['horizon_row_fraction'] * h
            fy = w / (2 * math.tan(math.radians(args.horizontal_fov_deg) / 2))
            pitch = math.degrees(math.atan(((h - 1) / 2 - row) / fy))
            rows.append(dict(frame=int(index), accepted=True, width=w, height=h,
                             horizon_row_px=row, pitch_deg=pitch, **record))
            pitches.append(pitch)
        capture.release()
        valid = [r for r in rows if r['accepted']]
        values = [r['pitch_deg'] for r in valid]
        summaries.append(dict(path=path, samples=rows, accepted=len(valid),
                              median_pitch_deg=float(np.median(values)) if values else None,
                              variance_pitch_deg2=float(np.var(values)) if values else None,
                              median_horizon_row_fraction=float(np.median([r['horizon_row_fraction'] for r in valid])) if valid else None,
                              variance_horizon_row_fraction=float(np.var([r['horizon_row_fraction'] for r in valid])) if valid else None))
    if not pitches:
        raise RuntimeError('no vanishing point consensus; no pitch estimate written')
    result = dict(calibrated=False, source='orchard_hough_vanishing_point_proxy',
                  horizontal_fov_deg=args.horizontal_fov_deg,
                  median_pitch_deg=float(np.median(pitches)), variance_pitch_deg2=float(np.var(pitches)),
                  std_pitch_deg=float(np.std(pitches)),
                  median_absolute_deviation_pitch_deg=float(np.median(np.abs(pitches - np.median(pitches)))),
                  assumptions=['horizontal FOV assumed, not identifiable from these videos',
                               'parallel level orchard rows; zero roll; centred principal point; square pixels',
                               'video camera height unknown; only pitch proxy used; robot height remains 0.275m',
                               'line consensus can fit vegetation and is not a calibrated horizon'], videos=summaries)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'videos'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
