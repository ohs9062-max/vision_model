import json
from pathlib import Path

from PIL import Image, ImageDraw


# ============================================================
# 경로
# ============================================================

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "images"
SEG_DIR = ROOT / "segmentation"
DET_DIR = ROOT / "detection"
META_DIR = ROOT / "metadata"
DRAW_DIR = ROOT / "draw"

for directory in [
    IMAGE_DIR,
    SEG_DIR,
    DET_DIR,
    META_DIR,
    DRAW_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 클래스 정의
# ============================================================

SEG_VALUE = {
    "drivable": 0,
    "caution": 1,
    "non_drivable": 2,
}

SEG_NAME = {
    0: "drivable",
    1: "caution",
    2: "non_drivable",
}


# ============================================================
# Segmentation 우선순위
# ============================================================
#
# 작은 숫자부터 먼저 그린다.
#
# 최종 결과:
#
# local non_drivable
#   > caution
#   > drivable
#
# 단,
# 전체 화면을 덮는 배경용
# non_drivable rectangle은
# 별도로 감지해서 그리지 않는다.
#
# 기본 mask 자체가 이미
# non_drivable(2)이기 때문이다.
# ============================================================

SEG_PRIORITY = {
    "drivable": 10,
    "caution": 20,
    "non_drivable": 30,
}


# ============================================================
# Detection 클래스
# ============================================================

DET_CLASS = {
    "step": 0,
    "ditch_hole": 1,
    "puddle": 2,
    "obstacle": 3,
}

DET_NAME = {
    0: "step",
    1: "ditch_hole",
    2: "puddle",
    3: "obstacle",
}


# ============================================================
# Overlay 색상
# ============================================================

SEG_COLOR = {
    "drivable": (
        76,
        210,
        71,
    ),
    "caution": (
        255,
        193,
        7,
    ),
    "non_drivable": (
        255,
        59,
        48,
    ),
}

DET_COLOR = {
    "step": (
        180,
        100,
        220,
    ),
    "ditch_hole": (
        35,
        105,
        225,
    ),
    "puddle": (
        25,
        200,
        215,
    ),
    "obstacle": (
        255,
        140,
        45,
    ),
}


# ============================================================
# 이미지 찾기
# ============================================================

def find_image(stem):
    """
    jpg / jpeg / png 모두 대응.
    """

    for suffix in [
        ".jpg",
        ".jpeg",
        ".png",
    ]:
        path = (
            IMAGE_DIR
            / f"{stem}{suffix}"
        )

        if path.exists():
            return path

    return None


# ============================================================
# Shape → Polygon
# ============================================================

def to_polygon(
    points,
    shape_type,
):
    pts = [
        (
            float(x),
            float(y),
        )
        for x, y in points
    ]

    if (
        shape_type == "rectangle"
        and len(pts) == 2
    ):
        x1, y1 = pts[0]
        x2, y2 = pts[1]

        pts = [
            (
                x1,
                y1,
            ),
            (
                x2,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                x1,
                y2,
            ),
        ]

    return pts


# ============================================================
# 배경용 non_drivable 판정
# ============================================================

def is_background_non_drivable(
    shape,
    width,
    height,
):
    """
    화면 전체를 덮기 위해 만든
    non_drivable rectangle인지 판단한다.

    이런 Shape는 실제 local hazard가 아니라
    '배경' 용도이므로 segmentation을
    그릴 때 제외한다.

    기본 mask 자체가 이미
    non_drivable(2)이기 때문이다.

    현재 프로젝트에서 사용했던
    전체 화면 rectangle처럼 좌표가
    이미지 경계를 약간 넘는 경우도 허용한다.
    """

    if (
        shape.get("label")
        != "non_drivable"
    ):
        return False

    shape_type = shape.get(
        "shape_type",
        "polygon",
    )

    # 배경용으로 사용했던
    # 전체 화면 rectangle만 특별 처리한다.
    if shape_type != "rectangle":
        return False

    points = shape.get(
        "points",
        [],
    )

    if len(points) < 2:
        return False

    xs = [
        float(point[0])
        for point in points
    ]

    ys = [
        float(point[1])
        for point in points
    ]

    x1 = max(
        0.0,
        min(xs),
    )

    y1 = max(
        0.0,
        min(ys),
    )

    x2 = min(
        float(width),
        max(xs),
    )

    y2 = min(
        float(height),
        max(ys),
    )

    box_width = max(
        0.0,
        x2 - x1,
    )

    box_height = max(
        0.0,
        y2 - y1,
    )

    width_ratio = (
        box_width
        / float(width)
    )

    height_ratio = (
        box_height
        / float(height)
    )

    # 이미지 폭/높이의 95% 이상을
    # 모두 덮으면 배경 rectangle으로 판단.
    return (
        width_ratio >= 0.95
        and height_ratio >= 0.95
    )


# ============================================================
# Segmentation 생성
# ============================================================

def make_segmentation(
    data,
    width,
    height,
):
    """
    최종 우선순위:

    1. 기본 배경 = non_drivable
    2. drivable
    3. caution
    4. local non_drivable

    전체 화면 배경용
    non_drivable rectangle은
    기본 mask와 역할이 같으므로 제외한다.

    따라서 다음처럼 겹쳐 그릴 수 있다.

    큰 drivable road
    +
    작은 caution puddle
    +
    작은 non_drivable obstacle

    최종 mask에서는
    local non_drivable이 가장 강하다.
    """

    # 기본 화면 전체를
    # non_drivable로 시작.
    mask = Image.new(
        "L",
        (
            width,
            height,
        ),
        SEG_VALUE[
            "non_drivable"
        ],
    )

    draw = ImageDraw.Draw(
        mask
    )

    shapes = []

    for shape in data.get(
        "shapes",
        [],
    ):
        label = shape.get(
            "label"
        )

        if label not in SEG_VALUE:
            continue

        # 전체 화면을 덮는
        # 배경용 non_drivable은 제외.
        if is_background_non_drivable(
            shape,
            width,
            height,
        ):
            continue

        shapes.append(
            shape
        )

    # 작은 우선순위부터 그린다.
    #
    # drivable
    # → caution
    # → local non_drivable
    shapes.sort(
        key=lambda shape:
        SEG_PRIORITY[
            shape["label"]
        ]
    )

    for shape in shapes:
        label = shape[
            "label"
        ]

        points = to_polygon(
            shape.get(
                "points",
                [],
            ),
            shape.get(
                "shape_type",
                "polygon",
            ),
        )

        if len(points) < 3:
            print(
                f"  [WARN] "
                f"{label}: "
                "polygon point 부족"
            )
            continue

        draw.polygon(
            points,
            fill=SEG_VALUE[
                label
            ],
        )

    return mask


# ============================================================
# Detection 생성
# ============================================================

def make_detection(
    data,
    width,
    height,
):
    """
    X-AnyLabeling Detection Shape를
    YOLO bbox로 변환한다.

    반환:
    - YOLO 문자열 목록
    - draw용 bbox 목록
    - Detection label 목록

    중요:
    obstacle bbox는 Detection이다.

    obstacle이 있다고 해서
    bbox 전체를 자동으로
    non_drivable로 만들지는 않는다.

    필요한 경우 X-AnyLabeling에서
    장애물의 실제 점유 영역에
    local non_drivable Shape를
    추가해야 한다.
    """

    lines = []
    boxes = []
    labels = []

    for shape in data.get(
        "shapes",
        [],
    ):
        label = shape.get(
            "label"
        )

        if label not in DET_CLASS:
            continue

        points = shape.get(
            "points",
            [],
        )

        if not points:
            continue

        xs = [
            float(point[0])
            for point in points
        ]

        ys = [
            float(point[1])
            for point in points
        ]

        x1 = max(
            0.0,
            min(xs),
        )

        y1 = max(
            0.0,
            min(ys),
        )

        x2 = min(
            float(width),
            max(xs),
        )

        y2 = min(
            float(height),
            max(ys),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            print(
                f"  [WARN] "
                f"{label}: "
                "잘못된 bbox"
            )
            continue

        cx = (
            (x1 + x2)
            / 2.0
            / width
        )

        cy = (
            (y1 + y2)
            / 2.0
            / height
        )

        bw = (
            (x2 - x1)
            / width
        )

        bh = (
            (y2 - y1)
            / height
        )

        class_id = (
            DET_CLASS[
                label
            ]
        )

        line = (
            f"{class_id} "
            f"{cx:.6f} "
            f"{cy:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f}"
        )

        lines.append(
            line
        )

        boxes.append(
            {
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

        labels.append(
            label
        )

    return (
        lines,
        boxes,
        labels,
    )


# ============================================================
# Metadata 갱신
# ============================================================

def update_metadata(
    stem,
    detection_labels,
):
    """
    기존 metadata가 있으면
    촬영정보는 그대로 유지한다.

    note만 현재 Detection 클래스 목록과
    일치시키도록 갱신한다.

    metadata가 없으면
    임의 생성하지 않는다.
    """

    path = (
        META_DIR
        / f"{stem}.json"
    )

    if not path.exists():
        return "NOT FOUND"

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

    except Exception as error:
        print(
            "  [WARN] "
            "metadata 읽기 실패: "
            f"{error}"
        )

        return "ERROR"

    metadata[
        "frame_id"
    ] = stem

    # Detection 클래스 순서 고정.
    ordered_labels = [
        name
        for name in DET_CLASS
        if name
        in detection_labels
    ]

    metadata[
        "note"
    ] = ordered_labels

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return "OK"


# ============================================================
# Draw Overlay
# ============================================================

def make_overlay(
    image,
    mask,
    boxes,
    output_path,
):
    base = image.convert(
        "RGBA"
    )

    width, height = (
        base.size
    )

    overlay = Image.new(
        "RGBA",
        (
            width,
            height,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    # --------------------------------------------------------
    # Segmentation Overlay
    # --------------------------------------------------------

    for class_id in [
        0,
        1,
        2,
    ]:
        label = SEG_NAME[
            class_id
        ]

        color = SEG_COLOR[
            label
        ]

        alpha_mask = mask.point(
            lambda value,
            cid=class_id:
            90
            if value == cid
            else 0
        )

        color_layer = Image.new(
            "RGBA",
            (
                width,
                height,
            ),
            (
                color[0],
                color[1],
                color[2],
                0,
            ),
        )

        color_layer.putalpha(
            alpha_mask
        )

        overlay = (
            Image.alpha_composite(
                overlay,
                color_layer,
            )
        )

    result = (
        Image.alpha_composite(
            base,
            overlay,
        )
    )

    draw = ImageDraw.Draw(
        result
    )

    # --------------------------------------------------------
    # Detection bbox
    # --------------------------------------------------------

    for box in boxes:
        label = box[
            "label"
        ]

        color = DET_COLOR[
            label
        ]

        x1 = int(
            box["x1"]
        )

        y1 = int(
            box["y1"]
        )

        x2 = int(
            box["x2"]
        )

        y2 = int(
            box["y2"]
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=color,
            width=4,
        )

        text = (
            f"GT {label}"
        )

        try:
            bbox = (
                draw.textbbox(
                    (
                        x1,
                        y1,
                    ),
                    text,
                )
            )

            text_width = (
                bbox[2]
                - bbox[0]
            )

            text_height = (
                bbox[3]
                - bbox[1]
            )

        except Exception:
            text_width = 80
            text_height = 12

        text_y = max(
            0,
            y1
            - text_height
            - 6,
        )

        draw.rectangle(
            [
                x1,
                text_y,
                x1
                + text_width
                + 6,
                text_y
                + text_height
                + 6,
            ],
            fill=(
                color[0],
                color[1],
                color[2],
                220,
            ),
        )

        draw.text(
            (
                x1 + 3,
                text_y + 3,
            ),
            text,
            fill=(
                255,
                255,
                255,
                255,
            ),
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_x = 15
    legend_y = 15
    line_height = 22

    legend_items = [
        (
            "GT Drivable",
            SEG_COLOR[
                "drivable"
            ],
        ),
        (
            "GT Caution",
            SEG_COLOR[
                "caution"
            ],
        ),
        (
            "GT Non-drivable",
            SEG_COLOR[
                "non_drivable"
            ],
        ),
        (
            "GT Step",
            DET_COLOR[
                "step"
            ],
        ),
        (
            "GT Ditch Hole",
            DET_COLOR[
                "ditch_hole"
            ],
        ),
        (
            "GT Puddle",
            DET_COLOR[
                "puddle"
            ],
        ),
        (
            "GT Obstacle",
            DET_COLOR[
                "obstacle"
            ],
        ),
    ]

    legend_height = (
        len(
            legend_items
        )
        * line_height
        + 10
    )

    draw.rectangle(
        [
            legend_x - 8,
            legend_y - 8,
            legend_x + 175,
            legend_y
            + legend_height,
        ],
        fill=(
            0,
            0,
            0,
            130,
        ),
    )

    for index, (
        text,
        color,
    ) in enumerate(
        legend_items
    ):
        y = (
            legend_y
            + index
            * line_height
        )

        draw.rectangle(
            [
                legend_x,
                y,
                legend_x + 12,
                y + 12,
            ],
            fill=(
                color[0],
                color[1],
                color[2],
                255,
            ),
        )

        draw.text(
            (
                legend_x + 18,
                y - 1,
            ),
            text,
            fill=(
                255,
                255,
                255,
                255,
            ),
        )

    result.convert(
        "RGB"
    ).save(
        output_path,
        quality=95,
    )


# ============================================================
# 검증
# ============================================================

def validate_mask(
    mask,
    stem,
):
    values = set(
        mask.getdata()
    )

    invalid = (
        values
        - {
            0,
            1,
            2,
        }
    )

    if invalid:
        print(
            f"  [ERROR] "
            f"{stem}: "
            "허용되지 않은 mask 값 "
            f"{sorted(invalid)}"
        )

        return False

    return True


def validate_detection(
    lines,
    stem,
):
    valid = True

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            print(
                f"  [ERROR] "
                f"{stem}: "
                "YOLO field 수 오류"
            )

            valid = False
            continue

        try:
            class_id = int(
                parts[0]
            )

            values = [
                float(value)
                for value
                in parts[1:]
            ]

        except ValueError:
            print(
                f"  [ERROR] "
                f"{stem}: "
                "YOLO 숫자 변환 실패"
            )

            valid = False
            continue

        if class_id not in [
            0,
            1,
            2,
            3,
        ]:
            print(
                f"  [ERROR] "
                f"{stem}: "
                "잘못된 class_id"
            )

            valid = False

        if any(
            value < 0.0
            or value > 1.0
            for value in values
        ):
            print(
                f"  [ERROR] "
                f"{stem}: "
                "YOLO 좌표 범위 오류"
            )

            valid = False

    return valid


# ============================================================
# Annotation 검색
# ============================================================

annotations = sorted(
    IMAGE_DIR.glob(
        "frame_*.json"
    )
)

if not annotations:
    print(
        "ERROR: "
        "golden_set/images 안에 "
        "X-AnyLabeling JSON이 없습니다."
    )

    raise SystemExit(1)


# ============================================================
# 실행
# ============================================================

print()

print(
    "Golden Set packing start"
)

print(
    "=" * 60
)

processed = 0
missing_image = 0
missing_metadata = 0
warning_count = 0
background_non_count = 0


for annotation_path in annotations:
    stem = (
        annotation_path.stem
    )

    image_path = (
        find_image(
            stem
        )
    )

    if image_path is None:
        print(
            f"[SKIP] "
            f"{stem}: "
            "원본 이미지 없음"
        )

        missing_image += 1
        continue

    try:
        with open(
            annotation_path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

    except Exception as error:
        print(
            f"[ERROR] "
            f"{stem}: "
            "annotation 읽기 실패 "
            f"{error}"
        )

        continue

    # X-AnyLabeling JSON인지 확인.
    if "shapes" not in data:
        print(
            f"[SKIP] "
            f"{stem}: "
            "X-AnyLabeling JSON 아님"
        )

        continue

    # --------------------------------------------------------
    # 알 수 없는 라벨 검사
    # --------------------------------------------------------

    unknown_labels = sorted(
        {
            shape.get(
                "label"
            )
            for shape
            in data.get(
                "shapes",
                [],
            )
            if shape.get(
                "label"
            )
            not in SEG_VALUE
            and shape.get(
                "label"
            )
            not in DET_CLASS
        }
    )

    if unknown_labels:
        print(
            f"  [WARN] "
            f"{stem}: "
            "지원되지 않는 label "
            f"{unknown_labels}"
        )

        warning_count += 1

    # --------------------------------------------------------
    # 실제 이미지 크기 사용
    # --------------------------------------------------------

    with Image.open(
        image_path
    ) as source:
        image = source.convert(
            "RGB"
        )

    width, height = (
        image.size
    )

    json_width = data.get(
        "imageWidth"
    )

    json_height = data.get(
        "imageHeight"
    )

    if (
        json_width
        and int(
            json_width
        )
        != width
    ):
        print(
            f"  [WARN] "
            f"{stem}: "
            "JSON imageWidth와 "
            "실제 이미지 폭이 다름"
        )

        warning_count += 1

    if (
        json_height
        and int(
            json_height
        )
        != height
    ):
        print(
            f"  [WARN] "
            f"{stem}: "
            "JSON imageHeight와 "
            "실제 이미지 높이가 다름"
        )

        warning_count += 1

    # --------------------------------------------------------
    # 배경용 non_drivable 개수 확인
    # --------------------------------------------------------

    frame_background_non = 0

    for shape in data.get(
        "shapes",
        [],
    ):
        if (
            is_background_non_drivable(
                shape,
                width,
                height,
            )
        ):
            frame_background_non += 1

    background_non_count += (
        frame_background_non
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    mask = make_segmentation(
        data,
        width,
        height,
    )

    seg_path = (
        SEG_DIR
        / f"{stem}.png"
    )

    mask.save(
        seg_path
    )

    mask_ok = (
        validate_mask(
            mask,
            stem,
        )
    )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    (
        detection_lines,
        boxes,
        detection_labels,
    ) = make_detection(
        data,
        width,
        height,
    )

    det_path = (
        DET_DIR
        / f"{stem}.txt"
    )

    det_path.write_text(
        "\n".join(
            detection_lines
        )
        + (
            "\n"
            if detection_lines
            else ""
        ),
        encoding="utf-8",
    )

    det_ok = (
        validate_detection(
            detection_lines,
            stem,
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_status = (
        update_metadata(
            stem,
            detection_labels,
        )
    )

    if (
        metadata_status
        == "NOT FOUND"
    ):
        missing_metadata += 1

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    draw_path = (
        DRAW_DIR
        / (
            f"{stem}"
            "_GT_overlay.jpg"
        )
    )

    make_overlay(
        image,
        mask,
        boxes,
        draw_path,
    )

    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    status = "OK"

    if (
        not mask_ok
        or not det_ok
    ):
        status = "CHECK"

    print(
        f"[{status}] "
        f"{stem} | "
        f"metadata="
        f"{metadata_status} | "
        f"detection="
        f"{len(detection_lines)} | "
        f"background_non="
        f"{frame_background_non}"
    )

    processed += 1


# ============================================================
# 전체 개수 확인
# ============================================================

image_count = len(
    [
        path
        for path
        in IMAGE_DIR.iterdir()
        if path.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    ]
)

annotation_count = len(
    list(
        IMAGE_DIR.glob(
            "frame_*.json"
        )
    )
)

seg_count = len(
    list(
        SEG_DIR.glob(
            "frame_*.png"
        )
    )
)

det_count = len(
    list(
        DET_DIR.glob(
            "frame_*.txt"
        )
    )
)

meta_count = len(
    list(
        META_DIR.glob(
            "frame_*.json"
        )
    )
)

draw_count = len(
    list(
        DRAW_DIR.glob(
            "frame_*_GT_overlay.jpg"
        )
    )
)


# ============================================================
# 완료 출력
# ============================================================

print()

print(
    "=" * 60
)

print(
    "완료"
)

print()

print(
    "처리한 annotation : "
    f"{processed}"
)

print(
    "이미지 없음       : "
    f"{missing_image}"
)

print(
    "metadata 없음     : "
    f"{missing_metadata}"
)

print(
    "경고              : "
    f"{warning_count}"
)

print(
    "배경 non 제외     : "
    f"{background_non_count}"
)

print()

print(
    "images            : "
    f"{image_count}"
)

print(
    "xany json         : "
    f"{annotation_count}"
)

print(
    "segmentation      : "
    f"{seg_count}"
)

print(
    "detection         : "
    f"{det_count}"
)

print(
    "metadata          : "
    f"{meta_count}"
)

print(
    "draw              : "
    f"{draw_count}"
)

print()

print(
    "dataset path      : "
    f"{ROOT}"
)
