# 마스크와 박스, 범례를 원본 옆의 예측 패널에 그린다.
"""Golden-set BGR palette/legend and original|Pred two-panel layout."""

import cv2
import numpy as np
from .postprocess import SEG_COLORS, BOX_COLORS, CLASS_NAMES


# 원본 옆에 예측 마스크·박스·범례를 배치한다.
def draw(image, result, *, legend=True):
    # 1) 세그멘테이션 색상과 예측 패널 만들기
    mask = result["mask"]
    if mask.dtype == np.uint8 and image.dtype == np.uint8:
        lut = np.zeros((256, 3), dtype=np.uint8)
        for c, v in SEG_COLORS.items():
            lut[c] = v
        color = lut[mask]
    else:
        color = np.zeros_like(image)
        for c, v in SEG_COLORS.items():
            color[mask == c] = v
    panel = cv2.addWeighted(image, 0.65, color, 0.35, 0)
    h, w = image.shape[:2]
    # 2) 박스와 클래스 점수 그리기
    for box, score, c in result["boxes"]:
        x1, y1, x2, y2 = [
            int(np.clip(round(v), 0, w - 1 if i % 2 == 0 else h - 1))
            for i, v in enumerate(box)
        ]
        col = BOX_COLORS[c]
        cv2.rectangle(panel, (x1, y1), (x2, y2), col, 3)
        text = f"Pred {CLASS_NAMES[c]} {score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ly1 = max(0, y1 - th - base - 8)
        ly2 = ly1 + th + base + 8
        cv2.rectangle(panel, (x1, ly1), (min(w - 1, x1 + tw + 10), ly2), col, -1)
        cv2.putText(
            panel,
            text,
            (x1 + 5, ly2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    # 3) 범례를 배치하고 원본과 연결하기
    if legend:
        names = [
            "Drivable",
            "Caution",
            "Non-drivable",
            "Step",
            "Ditch Hole",
            "Puddle",
            "Obstacle",
        ]
        colors = list(SEG_COLORS.values()) + list(BOX_COLORS.values())
        roi = panel[15:251, 15:296]
        if roi.size:
            cv2.addWeighted(np.zeros_like(roi), 0.35, roi, 0.65, 0, dst=roi)
        for i, (name, col) in enumerate(zip(names, colors)):
            y = 45 + i * 30
            cv2.rectangle(panel, (25, y - 15), (45, y + 5), col, -1)
            cv2.putText(
                panel,
                "Pred " + name,
                (55, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return np.hstack([image, panel])
