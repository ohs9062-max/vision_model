# ONNX Runtime 출력에 버전별 NumPy 후처리를 적용한다.
"""ONNX Runtime + NumPy/OpenCV only; no torch import."""

from pathlib import Path
import numpy as np
import yaml
import onnxruntime as ort
from .preprocess import preprocess, finish, cli
from .postprocess import decode_detector, Postprocessor
from .versioning import load_version


# 선택 버전의 ONNX 세션과 후처리를 제공한다.
class OnnxPredictor:
    # OnnxPredictor의 구성 요소와 설정을 초기화한다.
    def __init__(self, package=None, provider="CPUExecutionProvider", version=None):
        self.package = Path(package or Path(__file__).resolve().parents[1])
        self.version_dir, self.config, post_config = load_version(self.package, version)
        self.postprocessor = Postprocessor(post_config)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.sessions = {
            k: ort.InferenceSession(
                str(self.version_dir / f"{k}.onnx"),
                sess_options=options,
                providers=[provider],
            )
            for k in ("seg", "det")
        }

    # 두 ONNX 세션의 원출력을 가져온다.
    def outputs(self, image):
        return {
            k: self.sessions[k].run(
                None, {"images": preprocess(image, self.config[k]["input_size"])}
            )
            for k in ("seg", "det")
        }

    # 모델 원출력을 공통 후처리에 전달할 확률과 검출 기록으로 바꾼다.
    def raw(self, image):
        out = self.outputs(image)
        logits = out["seg"][0].astype(np.float32)
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs = (e / e.sum(1, keepdims=True))[0]
        rec = decode_detector(
            *out["det"],
            self.config["det"]["input_size"],
            getattr(self, "decode_threshold", self.config["det"].get("eval_threshold", 0.001)),
        )
        return probs, rec

    # 추론 출력에 버전별 후처리를 적용해 박스와 마스크를 반환한다.
    def __call__(self, image):
        probs, record = self.raw(image)
        return finish(
            probs,
            record,
            image.shape,
            self.config["det"]["input_size"],
            self.postprocessor,
        )


if __name__ == "__main__":
    cli(OnnxPredictor)
