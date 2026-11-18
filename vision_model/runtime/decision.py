# 비율 ROI 위험도와 연속 판정으로 주행 상태를 결정한다.
import cv2
import numpy as np


class Decision:
    # 판단 설정을 검사하고 안전한 초기 상태를 설정한다.
    def __init__(self, config):
        self.c = config
        roi = np.asarray(config['roi'], dtype=float)
        if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3 or not np.isfinite(roi).all() or ((roi < 0) | (roi > 1)).any():
            raise ValueError('roi must contain normalized polygon coordinates')
        for key in ('non_drivable_stop', 'caution_slow', 'obstacle_bottom_margin'):
            if not 0 <= config[key] <= 1:
                raise ValueError(key)
        if config['stable_frames'] < 1:
            raise ValueError('stable_frames must be positive')
        self.stop_frames = config.get('stop_frames', 1)
        self.go_frames = config.get('go_frames', config['stable_frames'])
        if min(self.stop_frames, self.go_frames) < 1:
            raise ValueError('stop_frames and go_frames must be positive')
        self.nonstop_count = 0
        self.state, self.pending, self.count = 'STOP', None, 0
        self.reasons = ['startup']

    # 입력·추론 오류는 즉시 정지하고 회복 카운터를 초기화한다.
    def fail(self, reason):
        self.state, self.pending, self.count = 'STOP', None, 0
        self.nonstop_count = 0
        self.reasons = [reason]
        return dict(decision='STOP', candidate='STOP', reasons=[reason], roi_ratios={})

    # 기존 공간 정책만 평가한다. 시간축 상태는 바꾸지 않는다.
    def evaluate(self, result):
        mask = result['mask']
        h, w = mask.shape
        polygon = np.rint(np.asarray(self.c['roi']) * [w - 1, h - 1]).astype(np.int32)
        roi = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi, [polygon], 1)
        area = int(roi.sum())
        if not area:
            raise ValueError('empty ROI')
        ratios = {name: float(np.count_nonzero((mask == i) & (roi != 0)) / area)
                  for i, name in enumerate(self.c['segmentation_classes'])}
        candidate, reasons = 'GO', []
        if ratios['non_drivable'] > self.c['non_drivable_stop']:
            candidate = 'STOP'
            reasons.append('non_drivable_roi')
        if ratios['caution'] > self.c['caution_slow']:
            if candidate != 'STOP':
                candidate = 'SLOW'
            reasons.append('caution_roi')
        for box, score, cid in result['boxes']:
            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:]).astype(int)
            if not roi[max(0,y1):max(0,min(h,y2)), max(0,x1):max(0,min(w,x2))].any():
                continue
            name = self.c['box_classes'][cid]
            if name == self.c['obstacle_class']:
                near = box[3] / h >= 1 - self.c['obstacle_bottom_margin']
                if near:
                    candidate = 'STOP'
                elif candidate != 'STOP':
                    candidate = 'SLOW'
                reasons.append('obstacle_near' if near else 'obstacle_roi')
            elif name in self.c['slow_classes']:
                if candidate != 'STOP':
                    candidate = 'SLOW'
                reasons.append(name + '_roi')
        return dict(candidate=candidate, candidate_reasons=reasons, roi_ratios=ratios)

    # 공간 정책 결과에 기존 히스테리시스와 정책 보호를 적용한다.
    def __call__(self, result):
        spatial = self.evaluate(result)
        candidate, reasons, ratios = spatial['candidate'], spatial['candidate_reasons'], spatial['roi_ratios']
        if candidate == self.pending:
            self.count += 1
        else:
            self.pending, self.count = candidate, 1
        self.nonstop_count = self.nonstop_count + 1 if candidate != 'STOP' else 0
        if candidate == 'STOP':
            ready = self.count >= self.stop_frames
        elif self.state == 'STOP' and (self.stop_frames != 1 or self.go_frames != self.c['stable_frames']):
            ready = self.nonstop_count >= self.go_frames
        else:
            ready = self.count >= self.go_frames if self.state == 'STOP' else self.count >= self.c['stable_frames']
        if ready:
            self.state, self.reasons = candidate, reasons
        if self.c.get('require_validated', False) and self.c.get('policy_validated') is not True:
            state = self.fail('policy_unvalidated')
            state.update(candidate=candidate, candidate_reasons=reasons, roi_ratios=ratios)
            return state
        held = self.state != candidate
        return dict(decision=self.state, candidate=candidate,
                    reasons=list(dict.fromkeys(self.reasons + (['stabilizing_' + candidate] if held else []))),
                    candidate_reasons=reasons, roi_ratios=ratios)
