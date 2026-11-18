# 입력 크기와 채널을 변환하고 원본 해상도 결과 및 CLI 출력을 만든다.
"""BGR uint8 HWC -> resized RGB float32 NCHW [0,1], no letterbox."""

import cv2
import numpy as np


# BGR 이미지를 지정 크기의 RGB FP32 텐서 배열로 바꾼다.
def preprocess(image, input_size):
    h, w = input_size
    resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (w, h))
    return np.ascontiguousarray(
        resized.transpose(2, 0, 1)[None], dtype=np.float32
    ) / np.float32(255)


# 검출 좌표 복원·영역 필터·박스 편향으로 최종 결과를 만든다.
def finish(probs, record, image_shape, det_size, postprocessor):
    oh, ow = image_shape[:2]
    # 11-13 POST_SCALE(선택 키, 기본 1.0=기존과 같음): 후처리를 축소 해상도에서 계산한 뒤 박스 좌표·마스크만 원본 크기로 되돌린다(CPU 시간 절감).
    scale = float(postprocessor.config.get("POST_SCALE", 1.0))
    w, h = (max(1, round(ow * scale)), max(1, round(oh * scale))) if scale < 1 else (ow, oh)
    base = np.stack(
        [cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(3)]
    )
    mask = base.argmax(0).astype(np.uint8)
    dets = postprocessor.rescue_record(record, mask, w, h, det_size[1], det_size[0])  # 11-09 BOX_RESCUE(없으면 기존과 같음)
    dets = postprocessor.drop_far_from_drivable(dets, mask)
    dets = postprocessor.keep_obstacle_touching_drivable(dets, mask)
    dets = postprocessor.drop_box_on_drivable(dets, mask)
    mask = postprocessor.apply_box_bias(base, dets, base_mask=mask)
    mask = postprocessor.clean_enclosed(mask)
    if (w, h) != (ow, oh):
        sx, sy = ow / w, oh / h
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        dets = [(type(d[0])([d[0][0] * sx, d[0][1] * sy, d[0][2] * sx, d[0][3] * sy]) if not isinstance(d[0], np.ndarray) else d[0] * np.array([sx, sy, sx, sy], dtype=d[0].dtype), *d[1:]) for d in dets]
    return {"boxes": dets, "mask": mask}


# 명령행 입력을 추론하고 박스·마스크·시각화 파일을 저장한다.
def cli(predictor_class):
    import argparse, json
    from pathlib import Path
    from .draw import draw

    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--package", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--device", default="cuda")
    p.add_argument("--provider", default="CPUExecutionProvider")
    a = p.parse_args()
    predictor = predictor_class(
        a.package,
        **(
            {"device": a.device}
            if predictor_class.__name__ == "TorchPredictor"
            else {"provider": a.provider}
        )
    )
    image = cv2.imread(a.image)
    if image is None:
        raise FileNotFoundError(a.image)
    result = predictor(image)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "boxes.json").write_text(json.dumps(result["boxes"], indent=2))
    if not cv2.imwrite(str(out / "mask.png"), result["mask"]):
        raise IOError(out)
    if not cv2.imwrite(str(out / "prediction.jpg"), draw(image, result)):
        raise IOError(out)
