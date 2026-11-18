#!/usr/bin/env python3
# crop_verifier ViT-S 체크포인트(M_all.pt)를 버전 폴더의 advice.ts(TorchScript)·advice.onnx·advice.json으로 내보낸다.
"""PC 전용(transformers 필요). AGX는 내보낸 파일만 쓴다(transformers 불필요). TensorRT 엔진은 AGX에서 build_trt.sh."""
import argparse, hashlib, inspect, json, sys
from pathlib import Path

import torch
from torch import nn

SIZE, BATCH = 224, 8
CLASSES = ["step", "ditch_hole", "puddle", "obstacle"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True)
        self.head = nn.Linear(768, 4)

    def forward(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))


def export(checkpoint, folder):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # M_all meta에는 classes가 없다(crop 학습 순서 = step, ditch_hole, puddle, obstacle). 있으면 확인한다.
    assert ck["meta"].get("classes", CLASSES) == CLASSES, ck["meta"].get("classes")
    net = Net(); net.load_state_dict(ck["model"]); net.eval()
    x = torch.randn(BATCH, 3, SIZE, SIZE)
    with torch.inference_mode():
        ref = net(x)
    ts = torch.jit.trace(net, x, check_trace=False); ts = torch.jit.freeze(ts.eval())
    ts.save(str(folder / "advice.ts"))
    back = torch.jit.load(str(folder / "advice.ts"))
    report = {}
    with torch.inference_mode():
        for n in (1, 3, BATCH, 13):
            y = torch.randn(n, 3, SIZE, SIZE)
            report[f"ts_batch{n}_maxdiff"] = float((back(y) - net(y)).abs().max())
    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(net, x, str(folder / "advice.onnx"), input_names=["images"], output_names=["logits"],
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, **kwargs)
    import onnx, onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(folder / "advice.onnx")))
    s = ort.InferenceSession(str(folder / "advice.onnx"), providers=["CPUExecutionProvider"])
    report["onnx_batch8_maxdiff"] = float(abs(s.run(None, {"images": x.numpy()})[0] - ref.numpy()).max())
    meta = dict(size=SIZE, batch=BATCH, mean=MEAN, std=STD, fill=[round(m * 255) for m in MEAN],
                classes=CLASSES, source=str(checkpoint),
                source_md5=hashlib.md5(Path(checkpoint).read_bytes()).hexdigest(), export=report)
    (folder / "advice.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--folder", required=True, help="weights/<버전> 폴더")
    a = p.parse_args()
    print(json.dumps(export(Path(a.checkpoint), Path(a.folder)), indent=2, ensure_ascii=False))
