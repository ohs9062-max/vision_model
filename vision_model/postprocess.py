import cv2
# 검출 박스 병합·제거와 세그멘테이션 편향을 버전별 설정으로 적용한다.
"""Version-specific NumPy postprocessing; no mutable global thresholds."""

import numpy as np

SEG_COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}

BOX_COLORS = {0: (255, 0, 255), 1: (255, 0, 0), 2: (255, 255, 0), 3: (0, 165, 255)}

CLASS_NAMES = {0: "step", 1: "ditch_hole", 2: "puddle", 3: "obstacle"}


# 11-09: 사슬 합치기 대신 쓰는 탐욕 합치기(MERGE_RULE mode=greedy). 점수 높은 박스부터 묶음을 키우되,
# 후보가 묶음 조각과 겹치고(간격 ≤gap), 합친 사각형 채움비 ≥fill, 합친 박스 ≤cap×화면일 때만 붙인다.
def _hull(bs):
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _union_area(bs):
    # 11-18 박스 많을 때 끊김(원시 후보 300개에서 프레임당 1.3s, 시간의 97%): 박스가 많으면 같은 면적을 numpy 격자로 계산
    if len(bs) > 16:
        return _union_area_np(bs)
    xs = sorted({b[0] for b in bs} | {b[2] for b in bs})
    total = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        spans = sorted((b[1], b[3]) for b in bs if b[0] < x1 and b[2] > x0)
        length, end = 0.0, -float("inf")
        for a, z in spans:
            length += max(0.0, z - max(a, end))
            end = max(end, z)
        total += (x1 - x0) * length
    return total


def _union_area_np(bs):
    """_union_area와 같은 합집합 면적(좌표 압축 격자 + 2차원 누적합). 값은 부동소수 덧셈 순서만 다르다."""
    b = np.asarray([[float(v) for v in q[:4]] for q in bs], dtype=np.float64)
    b = b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]
    if not len(b):
        return 0.0
    xs, ys = np.unique(b[:, [0, 2]]), np.unique(b[:, [1, 3]])
    xi0, xi1 = np.searchsorted(xs, b[:, 0]), np.searchsorted(xs, b[:, 2])
    yi0, yi1 = np.searchsorted(ys, b[:, 1]), np.searchsorted(ys, b[:, 3])
    d = np.zeros((len(xs) + 1, len(ys) + 1), np.int32)
    np.add.at(d, (xi0, yi0), 1); np.add.at(d, (xi1, yi0), -1); np.add.at(d, (xi0, yi1), -1); np.add.at(d, (xi1, yi1), 1)
    cov = d.cumsum(0).cumsum(1)[:-2, :-2] > 0
    return float(np.diff(xs) @ cov @ np.diff(ys))


def greedy_merge_groups(items, width, height, gap=0.0, fill=0.5, cap=0.25):
    rest = sorted(items, key=lambda d: -d[1])
    groups = []
    while rest:
        group = [rest.pop(0)]
        ug = [_union_area([group[0][0]])]  # 그룹 합집합 면적(누적)
        changed = True
        while changed:
            changed = False
            for d in list(rest):
                bs = [x[0] for x in group]
                if not any(
                    max(0, max(d[0][0], b[0]) - min(d[0][2], b[2])) <= gap * width
                    and max(0, max(d[0][1], b[1]) - min(d[0][3], b[3])) <= gap * height
                    for b in bs
                ):
                    continue
                nb = bs + [d[0]]
                h = _hull(nb)
                hull_area = (h[2] - h[0]) * (h[3] - h[1])
                if hull_area > cap * width * height and hull_area > max((b[2] - b[0]) * (b[3] - b[1]) for b in nb):
                    continue
                union = None
                if hull_area > 0:
                    # 11-18 속도(박스 많을 때 끊김): 합집합(그룹+d) = 그룹 합집합 + d 면적 − (d 안으로 자른 그룹 박스들의 합집합). 수학적으로 같은 값
                    a = d[0]
                    clipped = [[max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])] for b in bs]
                    clipped = [c for c in clipped if c[2] > c[0] and c[3] > c[1]]
                    union = ug[0] + (a[2] - a[0]) * (a[3] - a[1]) - (_union_area(clipped) if clipped else 0.0)
                    if union / hull_area < fill:
                        continue
                group.append(d)
                ug[0] = union if union is not None else _union_area(nb)
                rest.remove(d)
                changed = True
        groups.append(group)
    return groups


def drop_group_obstacles(dets, contain=None, min_count=None, area_ratio=None):
    """Drop enclosing obstacle boxes using the same unmodified input list."""
    if contain is None:
        return dets
    counts = []
    for i, (box, _, cls) in enumerate(dets):
        area = max(0., box[2]-box[0]) * max(0., box[3]-box[1])
        count = 0
        if cls == 3 and area > 0:
            for j, (small, _, other_cls) in enumerate(dets):
                if i == j or other_cls != 3:
                    continue
                small_area = max(0., small[2]-small[0]) * max(0., small[3]-small[1])
                if small_area <= 0 or small_area / area >= area_ratio:
                    continue
                inter = max(0., min(box[2],small[2])-max(box[0],small[0])) * max(0., min(box[3],small[3])-max(box[1],small[1]))
                if inter / small_area >= contain:
                    count += 1
        counts.append(count)
    return [d for d, count in zip(dets, counts) if count < min_count]


def keep_obstacle_touching_drivable(dets, seg_mask, ring=None, min_frac=None, classes=None):
    """Keep class 3 only when the expanded base-mask region has enough selected pixels."""
    if ring is None:
        return dets
    h, w = seg_mask.shape[:2]
    out = []
    for d in dets:
        if d[2] != 3:
            out.append(d); continue
        b = d[0]
        pad = max(4.0, ring * min(max(0., b[2]-b[0]), max(0., b[3]-b[1])))
        x1 = max(0, int(np.floor(b[0]-pad))); y1 = max(0, int(np.floor(b[1]-pad)))
        x2 = min(w, int(np.ceil(b[2]+pad))); y2 = min(h, int(np.ceil(b[3]+pad)))
        sub = seg_mask[y1:y2, x1:x2]
        if sub.size and np.count_nonzero(np.isin(sub, classes)) >= max(1, min_frac * sub.size):
            out.append(d)
    return out


def clean_enclosed(mask, config=None):
    """11-09: 화면 가장자리에 안 닿고 둘레(1px 고리)의 ring 이상이 non(2)인 caution 덩어리와
    화면 max_drivable 이하 drivable 덩어리를 non으로 바꾼다(non 안 drivable·caution = 라벨 실수, 사용자 결정)."""
    if not config:
        return mask
    h, w = mask.shape[:2]
    out = mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    for cls in (0, 1):
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask == cls).astype(np.uint8), connectivity=8)
        for j in range(1, n):
            x, y, bw, bh, area = stats[j]
            if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
                continue
            if cls == 0 and area > config["max_drivable"] * h * w:
                continue
            y0, x0 = y - 1, x - 1
            sub = (lab[y0:y + bh + 1, x0:x + bw + 1] == j).astype(np.uint8)
            ring = cv2.dilate(sub, kernel).astype(bool) & ~sub.astype(bool)
            if ring.any() and (mask[y0:y + bh + 1, x0:x + bw + 1][ring] == 2).mean() >= config["ring"]:
                out[y0:y + bh + 1, x0:x + bw + 1][sub.astype(bool)] = 2
    return out


def drop_box_on_drivable(dets, seg_mask, config=None):
    """Drop configured hazard boxes using the unbiased segmentation mask."""
    if not config or not config.get("enabled", False):
        return dets
    h, w = seg_mask.shape[:2]
    out = []
    for det in dets:
        if det[2] not in config["classes"]:
            out.append(det)
            continue
        x1, y1, x2, y2 = np.rint(det[0]).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        region = seg_mask[y1:y2, x1:x2]
        rule = config.get("class_rules", {}).get(det[2], config)  # 11-09 클래스별 기준(puddle 빛반사), 없으면 공통 기준
        if not region.size or not ((region == 1).mean() < rule["max_caution"] and (region == 0).mean() >= rule["min_drivable"]):
            out.append(det)
    return out


# 11-09: 임계 아래 조각 살리기(BOX_RESCUE). 임계 넘은 같은 클래스 박스에 간격 gap 이내로 맞닿고(이어서 사슬로)
# 박스 안 편향 전 세그 caution 비율이 min_caution 이상인 low 이상 조각만 살린다. dets는 합치기 전 박스.
def rescue_attach(dets, seg_mask, thresh, config=None):
    if not config:
        return [d for d in dets if d[1] >= thresh[d[2]]]
    h, w = seg_mask.shape[:2]
    g = config["gap"] * min(w, h)

    def caution(b):
        x1, y1, x2, y2 = np.rint(b).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        r = seg_mask[y1:y2, x1:x2]
        return (r == 1).mean() if r.size else 0.0

    def near(a, b):
        return not (a[2] + g < b[0] or b[2] + g < a[0] or a[3] + g < b[1] or b[3] + g < a[1])

    keep = [d for d in dets if d[1] >= thresh[d[2]]]
    cand = [d for d in dets if d[1] < thresh[d[2]] and d[2] in config["classes"] and d[1] >= config["low"] and caution(d[0]) >= config["min_caution"]]
    grown = True
    while grown:
        grown = False
        for d in list(cand):
            if any(k[2] == d[2] and near(k[0], d[0]) for k in keep):
                keep.append(d)
                cand.remove(d)
                grown = True
    return keep


# 버전별 임계값과 박스·마스크 후처리를 관리한다.
class Postprocessor:
    # Postprocessor의 구성 요소와 설정을 초기화한다.
    def __init__(self, config):
        self.config = config

    # 겹치는 장애물 박스를 묶어 최고 점수 박스만 남긴다.
    def dedup_obstacles(self, dets: list) -> list:
        """class 3 박스끼리 IoU >= 0.5 또는 (교집합 / 작은 박스 면적) >= 0.7이면 같은 물체로 판정하여
        연결 요소마다 점수 최고 박스 하나만 남김 (좌표는 키우지 않음)."""
        # 1) 장애물과 다른 검출 분리하기
        non_obs = [d for d in dets if d[2] != 3]
        obs = [d for d in dets if d[2] == 3]
        if len(obs) <= 1:
            return dets
        parent = list(range(len(obs)))

        # 연결 요소의 대표 인덱스를 찾고 경로를 압축한다.
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # 두 연결 요소를 같은 그룹으로 합친다.
        def union(x, y):
            rx, ry = (find(x), find(y))
            if rx != ry:
                parent[ry] = rx

        # 2) 겹침 조건으로 장애물 연결하기
        for i, (a, _, _) in enumerate(obs):
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            for j in range(i + 1, len(obs)):
                b = obs[j][0]
                area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = iw * ih
                if inter <= 0.0:
                    continue
                union_area = area_a + area_b - inter
                iou = inter / union_area if union_area > 0 else 0.0
                min_area = min(area_a, area_b)
                contain = inter / min_area if min_area > 0 else 0.0
                area_ratio = (
                    min_area / max(area_a, area_b) if max(area_a, area_b) > 0 else 0.0
                )
                if iou >= self.config["OBS_DUP_IOU"] or (
                    contain >= self.config["OBS_DUP_CONTAIN"]
                    and area_ratio >= self.config["OBS_DUP_AREA_RATIO"]
                ):
                    union(i, j)
        groups = {}
        for i, d in enumerate(obs):
            groups.setdefault(find(i), []).append(d)
        # 3) 연결 그룹마다 최고 점수 선택하기
        deduped_obs = [max(group, key=lambda d: d[1]) for group in groups.values()]
        return non_obs + deduped_obs

    # 주행 가능 영역에서 먼 지정 클래스 박스를 제거한다.
    def drop_far_from_drivable(
        self, dets: list, seg_mask: np.ndarray, margin: float = None
    ) -> list:
        """예측 세그 기준 주행 영역(drivable=0, caution=1)에서 먼 step·ditch·puddle 박스만 제거."""
        if margin is None:
            margin = self.config["FAR_MARGIN"]
        if not dets:
            return []
        # 1) 주행 영역 검사 범위 준비하기
        h, w = seg_mask.shape[:2]
        pad = margin * h
        out = []
        # 2) 박스 주변에 주행 영역이 있는지 검사하기
        for d in dets:
            if d[2] not in self.config["FAR_DROP_CLASSES"]:
                out.append(d)
                continue
            b = d[0]
            x1 = max(0, int(round(b[0] - pad)))
            y1 = max(0, int(round(b[1] - pad)))
            x2 = min(w, int(round(b[2] + pad)))
            y2 = min(h, int(round(b[3] + pad)))
            if x2 <= x1 or y2 <= y1:
                continue
            sub = seg_mask[y1:y2, x1:x2]
            if np.any((sub == 0) | (sub == 1)):
                out.append(d)
        return out

    def keep_obstacle_touching_drivable(self, dets, seg_mask):
        if "OBS_KEEP_RING" not in self.config:
            return dets
        return keep_obstacle_touching_drivable(dets, seg_mask, self.config["OBS_KEEP_RING"], self.config["OBS_KEEP_MIN_FRAC"], self.config["OBS_KEEP_CLASSES"])

    def clean_enclosed(self, mask):
        return clean_enclosed(mask, self.config.get("SEG_ENCLOSED_TO_NON"))

    def rescue_record(self, det_rec, seg_mask, width, height, det_w, det_h):
        """BOX_RESCUE가 있으면 낮은 임계로 원박스를 받아 조각 살리기 후 합치기, 없으면 기존 postprocess_record와 같다."""
        rule = self.config.get("BOX_RESCUE")
        if not rule:
            return self.postprocess_record(det_rec, width, height, det_w, det_h)
        thresh = self.config["POST_THRESH"]
        low = {c: (min(t, rule["low"]) if c in rule["classes"] else t) for c, t in thresh.items()}
        dets = self.postprocess_record(det_rec, width, height, det_w, det_h, thresh=low, merge=False)
        return self.postprocess(rescue_attach(dets, seg_mask, thresh, rule), width, height)

    def drop_box_on_drivable(self, dets, seg_mask):
        return drop_box_on_drivable(dets, seg_mask, self.config.get("DRIVABLE_BOX_DROP"))

    # 같은 클래스의 인접 박스를 병합하고 장애물 중복을 제거한다.
    def postprocess(self, dets: list, width: int, height: int) -> list:
        # 1) 병합하지 않는 클래스 보존하기
        out = [d for d in dets if d[2] not in self.config["MERGE_CLASSES"]]
        # 2) 클래스별 인접 박스를 연결하고 합치기
        rule = self.config.get("MERGE_RULE")
        for c in self.config["MERGE_CLASSES"]:
            items = [d for d in dets if d[2] == c]
            if rule is not None:
                for group in greedy_merge_groups(items, width, height, rule["gap"], rule["fill"], rule["cap"]):
                    out.append((_hull([d[0] for d in group]), max(d[1] for d in group), c))
                continue
            parent = list(range(len(items)))

            # 연결 요소의 대표 인덱스를 찾고 경로를 압축한다.
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            # 두 연결 요소를 같은 그룹으로 합친다.
            def union(x, y):
                x, y = (find(x), find(y))
                if x != y:
                    parent[y] = x

            for i, (a, _, _) in enumerate(items):
                for j in range(i + 1, len(items)):
                    b = items[j][0]
                    xgap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
                    ygap = max(0, max(a[1], b[1]) - min(a[3], b[3]))
                    if (
                        xgap <= self.config["MERGE_GAP"] * width
                        and ygap <= self.config["MERGE_GAP"] * height
                    ):
                        union(i, j)
            groups = {}
            for i, d in enumerate(items):
                groups.setdefault(find(i), []).append(d)
            for group in groups.values():
                bs, s = ([d[0] for d in group], max((d[1] for d in group)))
                out.append(
                    (
                        [
                            min((b[0] for b in bs)),
                            min((b[1] for b in bs)),
                            max((b[2] for b in bs)),
                            max((b[3] for b in bs)),
                        ],
                        s,
                        c,
                    )
                )
        out = self.dedup_obstacles(out)
        if "OBS_GROUP_CONTAIN" in self.config:
            out = drop_group_obstacles(out, self.config["OBS_GROUP_CONTAIN"], self.config["OBS_GROUP_MIN_COUNT"], self.config["OBS_GROUP_AREA_RATIO"])
        return out

    # 점수를 거른 검출 기록의 좌표를 원본 해상도로 복원한다.
    def postprocess_record(
        self,
        det_rec: dict,
        width: int,
        height: int,
        det_w: int,
        det_h: int,
        raw: bool = False,
        thresh: dict = None,
        merge: bool = True,
    ) -> list:
        """디텍터 원출력(입력 해상도 좌표) → 원본 픽셀 좌표 [(box, score, class)], raw가 아니면 채택 후처리 적용."""
        dets = []
        for b, score, label in zip(
            np.asarray(det_rec["boxes"]),
            np.asarray(det_rec["scores"]),
            np.asarray(det_rec["labels"]),
        ):
            class_id = int(label)
            if score < (
                self.config["DET_THRESH"]
                if raw
                else (thresh or self.config["POST_THRESH"])[class_id]
            ):
                continue
            dets.append(
                (
                    [
                        float(b[0]) * width / det_w,
                        float(b[1]) * height / det_h,
                        float(b[2]) * width / det_w,
                        float(b[3]) * height / det_h,
                    ],
                    float(score),
                    class_id,
                )
            )
        return dets if raw or not merge else self.postprocess(dets, width, height)

    # 검출 박스 안 클래스 확률에 편향을 더해 마스크를 결정한다.
    def apply_box_bias(
        self,
        seg_logits_or_probs: np.ndarray,
        dets: list,
        b_obs: float = None,
        b_cau: float = None,
        min_score: float = None,
        base_mask: np.ndarray = None,
    ) -> np.ndarray:
        """Apply detector-local class log-probability offsets and return the argmax mask.

        The input is a CxHxW probability map (or non-negative score map).  Only pixels
        inside retained detector boxes change; overlapping boxes add their offsets.
        """
        if b_obs is None:
            b_obs = self.config["BOX_BIAS"]["obstacle_non"]
        if b_cau is None:
            b_cau = self.config["BOX_BIAS"]["hazard_caution"]
        if min_score is None:
            min_score = self.config["BOX_BIAS"]["min_score"]
        # 1) 입력 확률을 검사하고 로그 확률 준비하기
        values = np.asarray(seg_logits_or_probs, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 3:
            raise ValueError("box bias expects a 3xHxW segmentation probability map")
        _, height, width = values.shape
        if base_mask is None:
            base_mask = values.argmax(axis=0).astype(np.uint8)
        regions = []
        # 2) 박스 영역의 대상 클래스에 편향 더하기
        for box, score, class_id in dets:
            if score < (self.config["POST_THRESH"][3] if class_id == 3 and "OBS_KEEP_RING" in self.config else min_score):
                continue
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1 = (max(0, x1), max(0, y1))
            x2, y2 = (min(width, x2), min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            if class_id == 3:
                regions.append((x1, y1, x2, y2, 2, b_obs))
            elif class_id in (0, 1, 2):
                regions.append((x1, y1, x2, y2, 1, b_cau))
        if not regions:
            return base_mask
        # Calculate logs only within the union bounding rectangle. This keeps
        # overlapping-box accumulation and the original argmax tie behavior.
        x1 = min(r[0] for r in regions)
        y1 = min(r[1] for r in regions)
        x2 = max(r[2] for r in regions)
        y2 = max(r[3] for r in regions)
        log_probs = np.log(np.maximum(values[:, y1:y2, x1:x2], 1e-12))
        for rx1, ry1, rx2, ry2, channel, bias in regions:
            log_probs[channel, ry1-y1:ry2-y1, rx1-x1:rx2-x1] += bias
        result = base_mask.copy()
        result[y1:y2, x1:x2] = log_probs.argmax(axis=0).astype(np.uint8)
        return result


# 검출 원출력을 점수·좌표로 변환하고 클래스별 NMS를 적용한다.
def decode_detector(logits, distances, priors, input_size, conf_threshold=0.001):
    """NumPy mirror of RTMDet.forward: multi-label sigmoid, topk, clip, NMS .65, 300."""
    # 1) 클래스 점수와 상위 후보 선택하기
    scores = 1 / (1 + np.exp(-logits[0]))
    ai, labels = np.where(scores >= conf_threshold)
    scores = scores[ai, labels]
    if len(scores) > 30000:
        top = np.argsort(-scores, kind="stable")[:30000]
        ai, labels, scores = ai[top], labels[top], scores[top]
    # 2) 좌표 복원 후 이미지 범위로 제한하기
    boxes = np.concatenate(
        (
            priors[None, :, :2] - distances[:, :, :2],
            priors[None, :, :2] + distances[:, :, 2:],
        ),
        axis=-1,
    )[0, ai].copy()
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, input_size[1])
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, input_size[0])
    valid = (boxes[:, 2:] > boxes[:, :2]).all(1)
    boxes, labels, scores = boxes[valid], labels[valid], scores[valid]
    # torchvision batched_nms coordinate trick, including float32 offsets.
    if len(boxes):
        shifted = boxes + labels.astype(boxes.dtype)[:, None] * (
            boxes.max() + np.float32(1)
        )
    else:
        shifted = boxes
    # 3) 클래스별 겹침을 억제하고 최대 검출 수 제한하기
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order) and len(keep) < 300:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        a = shifted[i]
        b = shifted[rest]
        inter = np.maximum(
            np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]), 0
        ).prod(1)
        union = (a[2:] - a[:2]).prod() + (b[:, 2:] - b[:, :2]).prod(1) - inter
        order = rest[inter / np.maximum(union, np.finfo(np.float32).tiny) <= 0.65]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
