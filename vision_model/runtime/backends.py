# 기존 원출력과 공통 후처리를 연결하고 TensorRT 10 실행을 제공한다.
import time
import numpy as np
from ..preprocess import preprocess, finish
from ..postprocess import Postprocessor, decode_detector
from ..versioning import load_version


class Engine:
    # TensorRT 10 이름 기반 텐서 API와 CUDA Runtime 바인딩을 준비한다.
    def __init__(self, path, trt, cuda):
        self.trt, self.cuda = trt, cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialize failed: " + str(path))
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT context creation failed")
        self.stream = self.check(cuda.cudaStreamCreate())[0]
        self.names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.buffers = {}

    # CUDA 에러 코드를 검사하고 반환값을 꺼낸다.
    @staticmethod
    def check(values):
        if int(values[0]) != 0:
            raise RuntimeError("CUDA runtime error: " + str(values[0]))
        return values[1:]

    # 정적 배치 엔진에 주소를 연결하고 비동기 추론 후 출력을 복사한다.
    def __call__(self, images, output_names, device_outputs=False):
        c, e, cuda = self.context, self.engine, self.cuda
        inputs = [
            n for n in self.names if e.get_tensor_mode(n) == self.trt.TensorIOMode.INPUT
        ]
        if inputs != ["images"]:
            raise ValueError("engine requires single images input")
        if not c.set_input_shape("images", images.shape):
            raise ValueError("TensorRT input shape rejected")
        hosts, outputs = {}, {}
        if device_outputs:
            import torch
        for name in self.names:
            shape = tuple(c.get_tensor_shape(name))
            if any(x <= 0 for x in shape):
                raise ValueError("unresolved TensorRT shape: " + name)
            dtype = np.dtype(self.trt.nptype(e.get_tensor_dtype(name)))
            if e.get_tensor_location(name) != self.trt.TensorLocation.DEVICE:
                raise ValueError("only device tensors supported")
            hosts[name] = (
                np.ascontiguousarray(images, dtype=dtype)
                if name == "images"
                else np.empty(shape, dtype)
            )
            a = hosts[name]
            if name not in self.buffers:
                self.buffers[name] = (
                    self.check(cuda.cudaMalloc(a.nbytes))[0],
                    a.nbytes,
                )
            ptr, size = self.buffers[name]
            if size != a.nbytes:
                raise ValueError("static engine shape changed")
            if name != "images" and device_outputs:
                dtype_torch = torch.from_numpy(np.empty((), dtype=dtype)).dtype
                outputs[name] = torch.empty(shape, device="cuda", dtype=dtype_torch)
                ptr = outputs[name].data_ptr()
            if not c.set_tensor_address(name, int(ptr)):
                raise RuntimeError("set_tensor_address failed: " + name)
        ptr = self.buffers["images"][0]
        self.check(
            cuda.cudaMemcpyAsync(
                ptr,
                hosts["images"].ctypes.data,
                hosts["images"].nbytes,
                cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
        )
        if not c.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        if not device_outputs:
            for name in output_names:
                a = hosts[name]
                self.check(
                    cuda.cudaMemcpyAsync(
                        a.ctypes.data,
                        self.buffers[name][0],
                        a.nbytes,
                        cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self.stream,
                    )
                )
        self.check(cuda.cudaStreamSynchronize(self.stream))
        return [outputs[n] if device_outputs else hosts[n] for n in output_names]

    # 소유한 CUDA 메모리와 스트림을 해제한다.
    def close(self):
        self.cuda.cudaStreamSynchronize(self.stream)
        for ptr, _ in self.buffers.values():
            self.cuda.cudaFree(ptr)
        self.buffers.clear()
        self.cuda.cudaStreamDestroy(self.stream)


def decode_detector_gpu(logits, distances, priors, input_size, conf_threshold):
    import torch

    # Candidate selection and coordinate reconstruction run on the device.
    score_map = torch.sigmoid(logits[0])
    ai, labels = torch.where(score_map >= conf_threshold)
    # Preserve the reference's NumPy sigmoid rounding for final scores.
    selected_logits = logits[0, ai, labels].cpu().numpy()
    scores = 1 / (1 + np.exp(-selected_logits))
    ai_cpu = ai.cpu().numpy()
    labels = labels.cpu().numpy()
    if len(scores) > 30000:
        top = np.argsort(-scores, kind="stable")[:30000]
        ai_cpu, labels, scores = ai_cpu[top], labels[top], scores[top]
    ai = torch.as_tensor(ai_cpu, device=distances.device)
    boxes = torch.cat(
        (
            priors[None, :, :2] - distances[:, :, :2],
            priors[None, :, :2] + distances[:, :, 2:],
        ),
        dim=-1,
    )[0, ai].clone()
    boxes[:, 0::2].clamp_(0, input_size[1])
    boxes[:, 1::2].clamp_(0, input_size[0])
    boxes = boxes.cpu().numpy()
    valid = (boxes[:, 2:] > boxes[:, :2]).all(1)
    boxes, labels, scores = boxes[valid], labels[valid], scores[valid]
    shifted = (
        boxes + labels.astype(boxes.dtype)[:, None] * (boxes.max() + np.float32(1))
        if len(boxes)
        else boxes
    )
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order) and len(keep) < 300:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        a, b = shifted[i], shifted[rest]
        inter = np.maximum(
            np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]), 0
        ).prod(1)
        union = (a[2:] - a[:2]).prod() + (b[:, 2:] - b[:, :2]).prod(1) - inter
        order = rest[inter / np.maximum(union, np.finfo(np.float32).tiny) <= 0.65]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


class TensorRTPredictor:
    # build_trt.sh의 버전별 seg.fp16.engine·det.fp16.engine을 연다.
    def __init__(self, version=None):
        try:
            import tensorrt as trt

            try:
                from cuda.bindings import runtime as cuda
            except ImportError:
                from cuda import cudart as cuda
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT backend requires TensorRT 10 and cuda-python runtime bindings on AGX; select torch/onnx on this PC"
            ) from exc
        if int(trt.__version__.split(".")[0]) != 10:
            raise RuntimeError("TensorRT 10 required; installed " + trt.__version__)
        self.version_dir, self.config, post = load_version(version=version)
        self.postprocessor = Postprocessor(post)
        self.decode_threshold = 0.10
        self.engines = {}
        try:
            for k in ("seg", "det"):
                self.engines[k] = Engine(
                    self.version_dir / f"{k}.fp16.engine", trt, cuda
                )
        except Exception:
            self.close()
            raise

    # 기존 ONNX와 같은 softmax·검출 디코드를 적용한다.
    def raw(self, image):
        logits = self.engines["seg"](
            preprocess(image, self.config["seg"]["input_size"]), ["logits"]
        )[0].astype(np.float32)
        det = self.engines["det"](
            preprocess(image, self.config["det"]["input_size"]),
            ["logits", "distances", "priors"],
        )
        if not np.isfinite(logits).all() or any(not np.isfinite(a).all() for a in det):
            raise FloatingPointError("nonfinite TensorRT output")
        e = np.exp(logits - logits.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True))[0], decode_detector(
            *det, self.config["det"]["input_size"], self.decode_threshold
        )

    def raw_gpu(self, image):
        import torch

        logits = self.engines["seg"](
            preprocess(image, self.config["seg"]["input_size"]), ["logits"], True
        )[0].float()
        det = self.engines["det"](
            preprocess(image, self.config["det"]["input_size"]),
            ["logits", "distances", "priors"],
            True,
        )
        if not bool(torch.isfinite(logits).all()) or any(
            not bool(torch.isfinite(a).all()) for a in det
        ):
            raise FloatingPointError("nonfinite TensorRT output")
        probs = torch.softmax(logits, dim=1)[0]
        record = decode_detector_gpu(
            *det, self.config["det"]["input_size"], self.decode_threshold
        )
        return probs, record

    def finish_gpu(self, probs, record, image_shape):
        import torch
        import torch.nn.functional as F

        h, w = image_shape[:2]
        det_size = self.config["det"]["input_size"]
        post = self.postprocessor
        base = F.interpolate(
            probs[None], size=(h, w), mode="bilinear", align_corners=False
        )[0]
        mask = base.argmax(0).byte().cpu().numpy()
        dets = post.rescue_record(record, mask, w, h, det_size[1], det_size[0])  # 11-09 BOX_RESCUE(없으면 기존과 같음)
        dets = post.drop_far_from_drivable(dets, mask)
        dets = post.keep_obstacle_touching_drivable(dets, mask)
        base_cpu = base.cpu().numpy()
        dets = post.drop_box_on_drivable(dets, mask)
        mask = post.apply_box_bias(base_cpu, dets, base_mask=mask)
        mask = post.clean_enclosed(mask)
        return {"boxes": dets, "mask": mask}

    # 공통 predictor 형식으로 기존 후처리 결과를 반환한다.
    def __call__(self, image):
        probs, record = self.raw_gpu(image)
        return self.finish_gpu(probs, record, image.shape)

    # 두 엔진의 메모리를 해제한다.
    def close(self):
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()


class Backend:
    # 선택한 기존 predictor를 한 번 생성한다.
    def __init__(self, config):
        kind = config["type"]
        if kind == "torch":
            import torch
            from ..infer_torch import TorchPredictor

            torch.set_num_threads(config["threads"])
            self.predictor = TorchPredictor(
                device=config["device"], version=config["version"]
            )
        elif kind == "onnx":
            import onnxruntime as ort
            from ..infer_onnx import OnnxPredictor

            if config["provider"] not in ort.get_available_providers():
                raise RuntimeError("unavailable ONNX provider: " + config["provider"])
            self.predictor = OnnxPredictor(
                provider=config["provider"], version=config["version"]
            )
        elif kind == "tensorrt":
            self.predictor = TensorRTPredictor(version=config["version"])
        else:
            raise ValueError("unknown backend: " + kind)
        threshold = float(config.get("decode_threshold", 0.10))
        if (
            not 0
            <= threshold
            <= min(list(self.predictor.postprocessor.config["POST_THRESH"].values()) + [self.predictor.postprocessor.config.get("BOX_RESCUE", {}).get("low", 1.0)])
        ):
            raise ValueError(
                "decode_threshold must be between 0 and the lowest postprocessing threshold"
            )
        if kind == "torch":
            self.predictor.models["det"].conf_threshold = threshold
        else:
            self.predictor.decode_threshold = threshold

    # 기존 함수 호출 사이만 측정해 추론·후처리 지연을 분리한다.
    def __call__(self, image):
        p = self.predictor
        t = time.perf_counter()
        is_trt = isinstance(p, TensorRTPredictor)
        probs, record = p.raw_gpu(image) if is_trt else p.raw(image)
        raw_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        result = (
            p.finish_gpu(probs, record, image.shape)
            if is_trt
            else finish(
                probs,
                record,
                image.shape,
                p.config["det"]["input_size"],
                p.postprocessor,
            )
        )
        return result, dict(
            inference=raw_ms, postprocess=(time.perf_counter() - t) * 1000
        )

    # 백엔드가 소유한 명시적 자원을 해제한다.
    def close(self):
        if hasattr(self.predictor, "close"):
            self.predictor.close()
