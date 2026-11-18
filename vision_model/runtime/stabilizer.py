"""Causal spatial tracks and a shared runtime decision factory."""
from collections import deque
import numpy as np
from .decision import Decision


class Stabilizer:
    def __init__(self, config):
        self.enabled = config.get('enabled', False)
        self.iou = float(config.get('iou_threshold', 0.3))
        self.distance = float(config.get('center_distance', 0.5))
        self.n = int(config.get('confirm_n', 3))
        self.m = int(config.get('confirm_m', 2))
        self.hold = int(config.get('hold_frames', 2))
        self.switch = int(config.get('switch_frames', 3))
        self.vote = int(config.get('vote_frames', 5))
        if not 0 <= self.iou <= 1 or self.distance < 0 or not 1 <= self.m <= self.n or self.hold < 0 or min(self.switch, self.vote) < 1:
            raise ValueError('invalid stabilizer configuration')
        self.tracks = []

    def reset(self):
        self.tracks.clear()

    def update(self, boxes):
        if not self.enabled:
            return boxes
        boxes = [(np.asarray(b, dtype=float), float(s), int(c)) for b, s, c in boxes]
        pairs = []
        for i, track in enumerate(self.tracks):
            a = track['box']
            for j, (b, _, _) in enumerate(boxes):
                inter = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
                union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - inter
                iou = inter / union if union else 0
                scale = max(np.linalg.norm((np.maximum(a[2:] - a[:2], 0) + np.maximum(b[2:] - b[:2], 0)) / 2), 1e-9)
                distance = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2) / scale
                if iou >= self.iou or distance <= self.distance:
                    pairs.append((iou, -distance, i, j))
        matched_t, matched_b = set(), set()
        for _, _, i, j in sorted(pairs, reverse=True):
            if i in matched_t or j in matched_b:
                continue
            matched_t.add(i); matched_b.add(j)
            t = self.tracks[i]
            b, score, cls = boxes[j]
            t['box'], t['score'], t['miss'] = b, score, 0
            t['hits'].append(1)
            t['votes'].append((cls, score))
            totals = {}
            for c, s in t['votes']:
                totals[c] = totals.get(c, 0) + s
            winner = max(totals, key=totals.get)
            if winner != t['class'] and totals[winner] > totals.get(t['class'], 0):
                if t['challenger'] == winner:
                    t['streak'] += 1
                else:
                    t['challenger'], t['streak'] = winner, 1
                if t['streak'] >= self.switch:
                    t['class'], t['challenger'], t['streak'] = winner, None, 0
            else:
                t['challenger'], t['streak'] = None, 0
            t['visible'] = t['visible'] or sum(t['hits']) >= self.m
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t['miss'] += 1
                t['hits'].append(0)
                t['challenger'], t['streak'] = None, 0
        self.tracks = [t for t in self.tracks if (t['visible'] and t['miss'] <= self.hold) or (not t['visible'] and t['miss'] < self.n and sum(t['hits']) > 0)]
        for j, (b, score, cls) in enumerate(boxes):
            if j not in matched_b:
                self.tracks.append(dict(box=b, score=score, **{'class': cls}, miss=0, hits=deque([1], maxlen=self.n), votes=deque([(cls, score)], maxlen=self.vote), visible=self.m == 1, challenger=None, streak=0))
        return [(t['box'].tolist(), t['score'], t['class']) for t in self.tracks if t['visible']]


def create_processors(config):
    return Stabilizer(config.get('stabilizer', {})), Decision(config['decision'])
