"""Temporary flat-ground geometry. Optical pixels -> REP-103 base_link metres.

Pitch is positive down. Intrinsics refer to the unrotated input raster; fx/fy/
cx/cy, when supplied, use reference_width/reference_height pixels. No lens
distortion, camera translation in x/y, yaw or roll correction is assumed.
"""
import math
from .config import require, number


class GroundGeometry:
    def __init__(self, config):
        keys = ('enabled', 'method', 'height_m', 'pitch_deg', 'horizontal_fov_deg',
                'max_range_m', 'robot_width_m', 'margin_m', 'stop_length_m',
                'slow_length_m', 'affect_candidate', 'source', 'table')
        require(config, keys, 'geometry')
        self.c = dict(config)
        self.enabled = self.c['enabled']
        self.method = self.c['method']
        for key in ('enabled', 'affect_candidate'):
            if type(self.c[key]) is not bool:
                raise ValueError('geometry.' + key + ' must be boolean')
        if not isinstance(self.c['source'], str) or not isinstance(self.c['table'], list):
            raise ValueError('geometry.source/table must be string/list')
        if self.method not in ('pinhole', 'table'):
            raise ValueError('geometry.method must be pinhole or table')
        for key in keys[2:10]:
            number(self.c[key], 'geometry.' + key)
        if min(self.c[k] for k in ('height_m', 'max_range_m', 'robot_width_m', 'stop_length_m')) <= 0:
            raise ValueError('geometry lengths must be positive')
        if self.c['margin_m'] < 0 or self.c['slow_length_m'] < self.c['stop_length_m']:
            raise ValueError('invalid geometry path zones')
        if not -89 < self.c['pitch_deg'] < 89 or not 0 < self.c['horizontal_fov_deg'] < 180:
            raise ValueError('invalid geometry angles')
        intrinsic_keys = ('fx', 'fy', 'cx', 'cy', 'reference_width', 'reference_height')
        if any(k in self.c for k in intrinsic_keys):
            if not all(k in self.c and math.isfinite(self.c[k]) for k in intrinsic_keys):
                raise ValueError('geometry intrinsics require fx/fy/cx/cy/reference_width/reference_height')
            if min(self.c[k] for k in ('fx', 'fy', 'reference_width', 'reference_height')) <= 0:
                raise ValueError('invalid geometry intrinsics')
        # Future row-distance table contract: original-row fraction and forward m.
        # No interpolation is enabled until lateral calibration is specified too.
        table = self.c['table']
        previous_row, previous_x = -1., math.inf
        for row, x in table:
            if not math.isfinite(row) or not math.isfinite(x) or not 0 <= row <= 1 or row <= previous_row or x <= 0 or x >= previous_x:
                raise ValueError('table rows must increase in [0,1], distances decrease and be positive')
            previous_row, previous_x = row, x

    def metadata(self):
        return dict(enabled=self.enabled, calibrated=False, method=self.method,
                    height_m=self.c['height_m'], pitch_deg=self.c['pitch_deg'],
                    source=self.c['source'])

    def intrinsics(self, width, height):
        if 'fx' in self.c:
            sx, sy = width / self.c['reference_width'], height / self.c['reference_height']
            return (self.c['fx'] * sx, self.c['fy'] * sy,
                    (self.c['cx'] + .5) * sx - .5, (self.c['cy'] + .5) * sy - .5)
        f = width / (2 * math.tan(math.radians(self.c['horizontal_fov_deg']) / 2))
        return f, f, (width - 1) / 2, (height - 1) / 2

    def project(self, u, v, width, height, rotate=0):
        """width/height describe the rotated image. Return x,y,reason (JSON safe)."""
        if rotate not in (0, 90, 180, 270):
            raise ValueError('invalid geometry rotation')
        if not self.enabled:
            return None, None, 'geometry_disabled'
        if self.method == 'table':
            return None, None, 'table_not_calibrated'
        if not all(math.isfinite(z) for z in (u, v, width, height)) or min(width, height) <= 0:
            return None, None, 'invalid_pixel'
        if not 0 <= u <= width - 1 or not 0 <= v <= height - 1:
            return None, None, 'pixel_outside_image'
        if rotate == 90:
            u, v, width, height = v, width - 1 - u, height, width
        elif rotate == 180:
            u, v = width - 1 - u, height - 1 - v
        elif rotate == 270:
            u, v, width, height = height - 1 - v, u, height, width
        fx, fy, cx, cy = self.intrinsics(width, height)
        q, lateral = (v - cy) / fy, -(u - cx) / fx
        pitch = math.radians(self.c['pitch_deg'])
        forward = math.cos(pitch) - q * math.sin(pitch)
        down = math.sin(pitch) + q * math.cos(pitch)
        if down <= 1e-10:
            return None, None, 'above_horizon'
        scale = self.c['height_m'] / down
        x, y = scale * forward, scale * lateral
        if x < 0:
            return None, None, 'behind_robot'
        if math.hypot(x, y) > self.c['max_range_m']:
            return None, None, 'beyond_max_range'
        return float(x), float(y), None

    def zone(self, x, left_y, right_y):
        if any(v is None for v in (x, left_y, right_y)):
            return 'none'
        half = self.c['robot_width_m'] / 2 + self.c['margin_m']
        if min(left_y, right_y) > half or max(left_y, right_y) < -half or x < 0:
            return 'none'
        if x <= self.c['stop_length_m']:
            return 'stop'
        return 'slow' if x <= self.c['slow_length_m'] else 'none'

    def hazards(self, boxes, classes, width, height, rotate=0):
        hazards = []
        for box, score, cid in boxes:
            # Boxes use half-open image edges. Map edges at W/H to last pixel.
            u1, u2 = max(0., min(width - 1., float(box[0]))), max(0., min(width - 1., float(box[2])))
            v = max(0., min(height - 1., float(box[3])))
            x, y, reason = self.project((u1 + u2) / 2, v, width, height, rotate)
            lx, ly, lr = self.project(u1, v, width, height, rotate)
            rx, ry, rr = self.project(u2, v, width, height, rotate)
            valid = reason is None and lr is None and rr is None
            # Rotated lower edges need not be iso-depth; use their nearest depth.
            zone = self.zone(min(x, lx, rx), ly, ry) if valid else 'none'
            hazards.append(dict(**{'class': classes[int(cid)]}, score=float(score),
                                x_m=x, y_m=y, left_y_m=ly, right_y_m=ry,
                                zone=zone, range_valid=valid, reason=reason or lr or rr,
                                frame_id='base_link'))
        return hazards
