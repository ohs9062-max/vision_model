# 검출 박스마다 crop 분류기(ViT-S, DINOv2-small)로 판정해, 분류기가 다른 클래스를 thresh 이상 확신하면 그 박스를 지운다(L2).
"""ADVICE (11-18): robot3 tools/vit_advice.py와 같은 규칙을 패키지 안에서 실행한다.

버전 폴더 파일: advice.json(전처리 상수·출처), advice.ts(torch, TorchScript), advice.onnx(onnx),
advice.fp16.engine(tensorrt, AGX에서 tools/build_trt.sh로 생성, 정적 배치 advice.json batch).
postprocess.yaml 의 ADVICE: {thresh, scale, classes} 가 있을 때만 켜진다. 박스 클래스·점수는 바꾸지 않는다.
조각: 박스 중심 기준 scale배 영역, 긴 변 224로 bicubic(antialias) 축소, 평균색 224 정사각 가운데 배치, ImageNet 정규화.
"""
import json

import numpy as np


class Advisor:
    def __init__(self, version_dir, config, kind):
        import torch

        self.torch = torch
        meta = json.loads((version_dir / "advice.json").read_text())
        self.size = int(meta["size"])
        self.batch = int(meta["batch"])
        self.thresh = float(config["thresh"])
        self.classes = {int(c) for c in config["classes"]}
        self.half = (float(config["scale"]) - 1) / 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = self.device
        self.fill = torch.tensor(meta["fill"], dtype=torch.float32, device=dev)[:, None, None].expand(3, self.size, self.size).contiguous()
        self.mean = torch.tensor(meta["mean"], dtype=torch.float32, device=dev)[:, None, None]
        self.std = torch.tensor(meta["std"], dtype=torch.float32, device=dev)[:, None, None]
        self.kind = kind
        if kind == "torch":
            self.net = torch.jit.load(str(version_dir / "advice.ts"), map_location=dev).eval()
        elif kind == "onnx":
            import onnxruntime as ort

            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
            self.session = ort.InferenceSession(str(version_dir / "advice.onnx"), providers=providers)
        elif kind == "tensorrt":
            import tensorrt as trt

            try:
                from cuda.bindings import runtime as cuda
            except ImportError:
                from cuda import cudart as cuda
            from .runtime.backends import Engine

            self.engine = Engine(version_dir / "advice.fp16.engine", trt, cuda)
        else:
            raise ValueError("unknown advice backend: " + kind)

    # 박스 하나를 조각 텐서(3,size,size)로 만든다. frame: (3,H,W) float RGB 0~255.
    def crop(self, frame, box):
        torch = self.torch
        H, W = frame.shape[1:]
        x1, y1, x2, y2 = [float(v) for v in box]
        px, py = (x2 - x1) * self.half, (y2 - y1) * self.half
        c = frame[:, max(0, int(y1 - py)):min(H, int(y2 + py)), max(0, int(x1 - px)):min(W, int(x2 + px))]
        if c.numel() == 0:
            c = frame[:, max(0, int(y1)):max(int(y1) + 1, int(y2)), max(0, int(x1)):max(int(x1) + 1, int(x2))]
        h, w = c.shape[1:]
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        r = torch.nn.functional.interpolate(c[None], size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)[0].round().clamp(0, 255)
        out = self.fill.clone()
        oy, ox = (self.size - nh) // 2, (self.size - nw) // 2
        out[:, oy:oy + nh, ox:ox + nw] = r
        return (out / 255.0 - self.mean) / self.std

    def logits(self, x):
        torch = self.torch
        if self.kind == "torch":
            with torch.inference_mode():
                return self.net(x).float()
        out = []
        for i in range(0, len(x), self.batch):
            chunk = x[i:i + self.batch]
            n = len(chunk)
            if self.kind == "onnx":
                out.append(torch.from_numpy(self.session.run(None, {"images": chunk.cpu().numpy()})[0]))
            else:
                pad = torch.zeros((self.batch,) + tuple(chunk.shape[1:]), dtype=chunk.dtype, device=chunk.device)
                pad[:n] = chunk
                out.append(torch.from_numpy(self.engine(pad.cpu().numpy(), ["logits"])[0][:n].astype(np.float32)))
        return torch.cat(out).float()

    # image: BGR uint8, boxes: [(xyxy, score, cls), ...] → (N,4) 확률
    def probs(self, image, boxes):
        torch = self.torch
        if not len(boxes):
            return np.zeros((0, 4), np.float32)
        with torch.inference_mode():
            frame = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).to(self.device).permute(2, 0, 1).float()
            x = torch.stack([self.crop(frame, b[0]) for b in boxes])
            return self.logits(x).softmax(1).cpu().numpy()

    # L2: 분류기 argmax가 박스 클래스와 다르고 그 확률이 thresh 이상이면 제거
    def apply(self, image, boxes):
        boxes = list(boxes)
        idx = [i for i, b in enumerate(boxes) if int(b[2]) in self.classes]
        if not idx:
            return boxes
        p = self.probs(image, [boxes[i] for i in idx])
        drop = {i for i, pi in zip(idx, p) if int(np.argmax(pi)) != int(boxes[i][2]) and float(pi.max()) >= self.thresh}
        return [b for i, b in enumerate(boxes) if i not in drop]

    def close(self):
        if self.kind == "tensorrt":
            self.engine.close()


def validate_advice(config):
    if (
        not isinstance(config, dict)
        or set(config) != {"thresh", "scale", "classes"}
        or type(config["thresh"]) not in (int, float)
        or not 0 < config["thresh"] <= 1
        or type(config["scale"]) not in (int, float)
        or not 1 <= config["scale"] <= 2
        or not isinstance(config["classes"], list)
        or any(type(c) is not int or c not in range(4) for c in config["classes"])
    ):
        raise ValueError("Invalid ADVICE")
