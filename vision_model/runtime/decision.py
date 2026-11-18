# 비율 ROI 위험도와 연속 판정으로 주행 상태를 결정한다.
import cv2
import numpy as np
from .geometry import GroundGeometry
from .config import require, number, validate_rules


class Decision:
    # 판단 설정을 검사하고 안전한 초기 상태를 설정한다.
    def __init__(self, config):
        self.c = config
        require(config, ('geometry', 'roi', 'stable_frames', 'stop_frames', 'go_frames', 'policy_validated', 'require_validated', 'segmentation_classes', 'box_classes'), 'decision')
        validate_rules(config)
        self.geometry = GroundGeometry(config['geometry'])
        self.priority = {action: i for i, action in enumerate(config['action_priority'])}
        roi = np.asarray(config['roi'], dtype=float)
        if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3 or not np.isfinite(roi).all() or ((roi < 0) | (roi > 1)).any():
            raise ValueError('roi must contain normalized polygon coordinates')
        for key in ('stable_frames', 'stop_frames', 'go_frames'):
            number(config[key], 'decision.' + key, minimum=1, integer=True)
        for key in ('policy_validated', 'require_validated'):
            if type(config[key]) is not bool:
                raise ValueError('decision.' + key + ' must be boolean')
        self.stop_frames = config['stop_frames']
        self.go_frames = config['go_frames']
        self.nonstop_count = 0
        self.state, self.pending, self.count = 'STOP', None, 0
        self.reasons = ['startup']

    # 입력·추론 오류는 즉시 정지하고 회복 카운터를 초기화한다.
    def fail(self, reason):
        self.state, self.pending, self.count = 'STOP', None, 0
        self.nonstop_count = 0
        self.reasons = [reason]
        return dict(decision='STOP', candidate='STOP', reasons=[reason], roi_ratios={},
                    hazards=[], hazards_frame_id='base_link', geometry=self.geometry.metadata())

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

        def apply(action, reason):
            nonlocal candidate
            if self.priority[action] < self.priority[candidate]:
                candidate = action
            reasons.append(reason)

        for rule in self.c['segmentation_rules']:
            if ratios[rule['class']] > rule['ratio_gt']:
                apply(rule['action'], rule['reason'])
        for box, score, cid in result['boxes']:
            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:]).astype(int)
            if not roi[max(0,y1):max(0,min(h,y2)), max(0,x1):max(0,min(w,x2))].any():
                continue
            name = self.c['box_classes'][cid]
            for rule in self.c['box_rules']:
                if name != rule['class']:
                    continue
                near = ('near_bottom_margin' in rule and
                        box[3] / h >= 1 - rule['near_bottom_margin'])
                apply(rule['near_action'] if near else rule['action'],
                      rule['near_reason'] if near else rule['reason'])
        hazards = self.geometry.hazards(result['boxes'], self.c['box_classes'], w, h, result['rotate'] if 'rotate' in result else 0)
        if self.geometry.enabled and self.geometry.c['affect_candidate']:
            for hazard in hazards:
                for rule in self.c['geometry_rules']:
                    if hazard['zone'] == rule['zone']:
                        apply(rule['action'], rule['reason'])
        return dict(candidate=candidate, candidate_reasons=reasons, roi_ratios=ratios,
                    hazards=hazards, hazards_frame_id='base_link', geometry=self.geometry.metadata())

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
        if self.c['require_validated'] and self.c['policy_validated'] is not True:
            state = self.fail('policy_unvalidated')
            state.update(spatial)
            return state
        held = self.state != candidate
        return dict(decision=self.state,
                    reasons=list(dict.fromkeys(self.reasons + (['stabilizing_' + candidate] if held else []))),
                    **spatial)
