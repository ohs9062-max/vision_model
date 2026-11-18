"""Causal spatial tracks and a shared runtime decision factory."""
from collections import deque
import numpy as np
from .decision import Decision
from .config import require, number


class Stabilizer:
    def __init__(self, config):
        require(config, ('enabled', 'iou_threshold', 'center_distance', 'confirm_n', 'confirm_m', 'hold_frames', 'switch_frames', 'vote_frames'), 'stabilizer')
        if type(config['enabled']) is not bool:
            raise ValueError('stabilizer.enabled must be boolean')
        for key in ('iou_threshold', 'center_distance', 'confirm_n', 'confirm_m', 'hold_frames', 'switch_frames', 'vote_frames'):
            number(config[key], 'stabilizer.' + key, minimum=0,
                   integer=key not in ('iou_threshold', 'center_distance'))
        self.enabled = config['enabled']
        self.iou = config['iou_threshold']
        self.distance = config['center_distance']
        self.n = config['confirm_n']
        self.m = config['confirm_m']
        self.hold = config['hold_frames']
        self.switch = config['switch_frames']
        self.vote = config['vote_frames']
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
                totals[c] = (totals[c] if c in totals else 0) + s
            winner = max(totals, key=totals.get)
            if winner != t['class'] and totals[winner] > (totals[t['class']] if t['class'] in totals else 0):
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
    require(config, ('stabilizer', 'decision', 'geometry'), 'runtime')
    return Stabilizer(config['stabilizer']), Decision(dict(config['decision'], geometry=config['geometry']))
