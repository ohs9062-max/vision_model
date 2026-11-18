# 버전별 가중치와 후처리를 불러와 PyTorch AMP 추론을 수행한다.
from pathlib import Path
import numpy as np
import torch
from .versioning import load_version
from .postprocess import Postprocessor
from .models import build_cspnext_seg, RTMDet
from .preprocess import preprocess, finish, cli


# 선택 버전의 PyTorch 모델 추론을 제공한다.
class TorchPredictor:
    # TorchPredictor의 구성 요소와 설정을 초기화한다.
    def __init__(self, package=None, device="cuda", version=None):
        self.package = Path(package or Path(__file__).resolve().parents[1])
        self.device = torch.device(device)
        self.version_dir, _, post_config = load_version(self.package, version)
        self.postprocessor = Postprocessor(post_config)
        # 1) 버전별 모델 구성과 가중치 불러오기
        self.models = {}
        self.config = {}
        for kind in ("seg", "det"):
            c = torch.load(
                self.version_dir / f"{kind}.pt", map_location="cpu", weights_only=False
            )
            self.config[kind] = c["config"]
            model = (
                build_cspnext_seg(
                    3, class_weights=c["model_state"].get("class_weights")
                )
                if kind == "seg"
                else RTMDet(
                    num_classes=4, conf_thresh=c["config"].get("eval_threshold", 0.001)
                )
            )
            model.load_state_dict(c["model_state"], strict=True)
            self.models[kind] = model.to(self.device).eval()

    # 원본과 같은 연산 순서로 PyTorch 입력 텐서를 만든다.
    def tensor(self, image, kind):
        # Match original torch float().div(255), including operation order.
        import cv2

        h, w = self.config[kind]["input_size"]
        a = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (w, h))
        return (
            torch.from_numpy(a.transpose(2, 0, 1))
            .float()
            .div(255)
            .unsqueeze(0)
            .to(self.device)
        )

    # 모델 원출력을 공통 후처리에 전달할 확률과 검출 기록으로 바꾼다.
    @torch.inference_mode()
    def raw(self, image):
        x = self.tensor(image, "seg")
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits = self.models["seg"](x)
        if not torch.isfinite(logits).all():
            with torch.autocast(self.device.type, enabled=False):
                logits = self.models["seg"](x.float())
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite FP32 evaluation logits")
        # 2) 세그 확률과 검출 기록 만들기
        probs = logits.float().softmax(1)[0].cpu().numpy()
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            record = self.models["det"](self.tensor(image, "det"))[0]
        return probs, {k: v.cpu().numpy() for k, v in record.items()}

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
    cli(TorchPredictor)
