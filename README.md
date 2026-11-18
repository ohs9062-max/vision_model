## 가장 쉬운 사용법

`영상재생.sh`를 더블클릭하거나 `bash 영상재생.sh`를 실행합니다.
[영상 파일 열기]에서 영상을 고르고 [재생]을 누릅니다.
[정지]로 재생을 끝냅니다. 이동 버튼으로 5초 또는 30초 앞뒤로 이동하고 [⏯ 일시정지]로 멈추거나 다시 재생합니다. [↻ 화면 회전]은 누를 때마다 90도 회전하며 [느리게]와 [빠르게]는 배속을 0.25·0.5·1·2·4배로 조절합니다. 회전과 배속은 재생 전에 선택할 수도 있습니다. 현재 위치와 전체 길이가 표시됩니다.

# Weatherfar 현재 학생 배포 패키지

CSPNeXt-L 세그(288×512) + RTMDet-L 검출(640×640). 학습 저장소, mmdet/mmcv 없이 실행합니다. `weights/<active_model>/seg.pt`, `det.pt`는 원본 그대로 복사했으며 SHA256는 `weights/20261026_weatherfar/verify.json`에 있습니다. 버전별 모델 config는 `model_config.yaml`에 보존했습니다. 최상위 `config.yaml`의 `active_model`이 실행 버전을 선택합니다. 원본 .pt에는 optimizer 등 학습 상태도 포함되어 파일이 큽니다.

## Jetson AGX Orin 설치

현재 JetPack/L4T를 `head -n 1 /etc/nv_tegra_release`와 `dpkg-query -W nvidia-jetpack`으로 확인하십시오. 아래는 알려진 배포 버전의 조합이며 설치된 L4T와 NVIDIA 호환표가 우선합니다.

| JetPack | CUDA / TensorRT | torch·torchvision / ONNX Runtime |
|---|---|---|
| 5.1 | 11.4 / 8.5.2 | JP5용 aarch64 NVIDIA torch wheel과 일치하는 torchvision; 해당 CUDA/cuDNN용 ORT 빌드 |
| 6.0 | 12.2 / 8.6 | JP6.0용 NVIDIA torch/torchvision; CUDA12.2 및 해당 cuDNN과 맞는 ORT |
| 6.2·6.2.1·6.2.2 | 12.6 / 10.3 | 해당 JP용 NVIDIA torch/torchvision; CUDA12.6/cuDNN9용 ORT |

공식 근거: [JP5.1](https://docs.nvidia.com/jetson/jetpack/5.1/release-notes/index.html), [JP6.0](https://docs.nvidia.com/jetson/jetpack/6.0/release-notes/index.html), [JP6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62), [JP6.2.1](https://docs.nvidia.com/jetson/jetpack/6.2.1/introduction/index.html), [JP6.2.2](https://developer.nvidia.com/embedded/jetpack-sdk-622).

1. JetPack 제공 CUDA/cuDNN/TensorRT를 설치합니다. [NVIDIA PyTorch 설치 안내](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)에서 해당 JetPack·Python·aarch64 wheel을 선택합니다. torchvision도 그 torch와 맞춰 설치하고 `torchvision.ops.nms` 실행 여부를 확인합니다.
2. `python3 -m venv --system-site-packages .venv` 후 활성화하고 `pip install numpy opencv-python PyYAML onnx`를 실행합니다. `requirements.txt`는 의존성 목록입니다. Jetson에서 전체를 무조건 설치하면 torch가 다른 빌드로 바뀔 수 있으므로 NVIDIA wheel을 먼저 선택합니다.
3. ONNX CPU 경로에는 `onnxruntime`, GPU 경로에는 해당 JetPack aarch64용 `onnxruntime-gpu` 하나를 설치합니다. [ORT 공식 설치 안내의 Jetson 링크](https://onnxruntime.ai/docs/install/)에서 NVIDIA 빌드를 선택하거나 CUDA/cuDNN 버전에 맞춰 소스 빌드합니다. `ort.get_available_providers()`에 `CUDAExecutionProvider`가 있는지 확인합니다. x86 PC wheel은 사용할 수 없습니다.

이 PC 검증 환경: torch 2.14.0+cu130, NVIDIA GeForce RTX 4070 Ti, ONNX Runtime CPU. AGX 실기 설치·추론·TensorRT 엔진 검증은 수행하지 않았습니다.

## 사용

패키지 루트에서 실행합니다. 출력 폴더에는 원본 크기 `mask.png`(uint8, 0/1/2), `boxes.json`, 원본|Pred 2분할 `prediction.jpg`가 저장됩니다.

```bash
python -m vision_model.infer_torch --image /path/frame.jpg --output /path/result_torch --device cuda
python -m vision_model.infer_onnx --image /path/frame.jpg --output /path/result_onnx --provider CPUExecutionProvider
python -m vision_model.infer_onnx --image /path/frame.jpg --output /path/result_onnx_gpu --provider CUDAExecutionProvider
python tools/export_onnx.py
bash tools/build_trt.sh
```

`build_trt.sh [PACKAGE_ROOT] [VERSION]`는 AGX에서 `/usr/src/tensorrt/bin/trtexec`로 `seg.fp16.engine`, `det.fp16.engine`를 생성합니다. 다른 경로는 `TRTEXEC=...`로 지정하십시오. 엔진은 대상 Jetson/TensorRT 버전에서 생성하십시오. 이 도구는 엔진 빌더이며 Python TensorRT 10 실행은 아래 실시간 실행부가 제공합니다. TensorRT 출력에도 아래의 동일한 decode/후처리를 적용해야 합니다. 현재 TensorRT FP16 정확도·속도는 미측정입니다.

```python
import cv2
from vision_model.infer_torch import TorchPredictor
# ONNX 전용 설치에서는 from vision_model.infer_onnx import OnnxPredictor
predictor = TorchPredictor(device='cuda')
result = predictor(cv2.imread('/path/frame.jpg'))
mask, boxes = result['mask'], result['boxes']
```

입력은 OpenCV BGR uint8 H×W×3입니다. RGB로 바꿔 직접 resize(비율 유지/letterbox 없음) 후 float32 NCHW [0,1]을 사용합니다. 세그 내부만 BGR 0..255 변환 및 학습된 mean/std 정규화를 수행합니다. 검출에는 별도 mean/std 정규화를 추가하지 않습니다. 박스는 `([x1,y1,x2,y2], score, class_id)` 목록이며 원본 영상 픽셀 실수 좌표입니다. 검출 클래스 0=step, 1=ditch_hole, 2=puddle, 3=obstacle; 세그 0=drivable, 1=caution, 2=non_drivable.

PyTorch CUDA는 정본과 동일한 AMP(FP16), 세그 logits를 FP32 softmax합니다. 비유한 세그 logits에는 FP32 재추론을 수행합니다. HARD_CLS는 원본 `compute_loss`의 양성 타깃에만 영향을 주므로 추론 플래그는 필요 없습니다. 독립 모델에서 학습 손실은 제거했습니다.

## ONNX 인터페이스

opset 17, static batch=1, FP32 입력/가중치, NMS는 그래프 외부 NumPy입니다. 단일 입력명은 `images`입니다.

- 세그: 입력 [1,3,288,512], 출력 logits [1, 3, 288, 512]; FP32 softmax 후 원본 크기로 클래스별 bilinear resize합니다.
- 검출: 입력 [1,3,640,640], 출력 `logits` [1, 8400, 4], `distances` [1, 8400, 4], `priors` [8400, 3]. distances는 stride가 곱해진 l/t/r/b, priors는 x/y/stride입니다.
- `decode_detector`: sigmoid → score≥0.001 → pre-NMS top30000 → 입력 경계 clip/퇴화박스 제거 → 클래스별 IoU .65 NMS → 전체 top300. 이어서 채택 후처리를 적용합니다.

## 채택 후처리 (순서 고정)

`postprocess.py`는 NumPy만 사용합니다. 원본 크기 복원에는 `preprocess.finish`의 OpenCV bilinear resize가 사용됩니다.

| 순서 | 작업 | 상수/규칙 |
|---|---|---|
| 1 | 클래스 점수 필터 | step .30, ditch .45, puddle .35, obstacle .30 |
| 2 | step·ditch 연결 요소 병합 | 원래 박스 간 xgap≤.01W AND ygap≤.01H, 외접 박스·최고 점수 |
| 3 | obstacle 중복 제거 | IoU≥.5 OR (작은 면적 대비 교집합≥.7 AND 면적비≥.5); 요소 최고 점수 박스 유지 |
| 4 | 세그 확률맵 resize와 argmax | 클래스별 INTER_LINEAR, 원본 크기 |
| 5 | drop_far_from_drivable | step·ditch·puddle만, .02H padding 내 세그 0 또는 1이 없으면 제거; obstacle 제외 |
| 6 | apply_box_bias | log(max(prob,1e-12)); score≥.3 박스 안 obstacle→class2 +3, 나머지→class1 +2, 겹침 가산 |
| 7 | 마스크 argmax | uint8, SAM2/추가 채움 없음 |

렌더링은 golden_set draw_gt 색상표, 범례(.35 검정 오버레이), 원본 .65 + 색상 .35, 박스 두께 3을 사용하며 라벨 앞에 Pred를 붙입니다. 눈으로 확인하지 않았으며 크기/타입만 smoke에서 검사했습니다.

## 검증 결과

(b) 합격 기준: PyTorch FP32(autocast·TF32 끔)와 ONNX FP32를 비교하며, 원출력 최대 절대차를 기록한다.  
클래스·개수 동일, 박스 좌표 차 ≤1px, 마스크 일치 ≥99.9%를 모두 만족해야 새 버전을 승인한다.  
(a) 참고 기준: 배포 PyTorch AMP FP16 대비 ONNX FP32의 클래스·개수 동일, 박스 ≤3px, 마스크 ≥99.5%이며 미달은 경고만 남긴다.  
AGX TensorRT FP16도 기기에서 (a) 방식과 허용치로 확인한다. 원출력 절대차의 별도 허용치는 지정하지 않으며, 비유한 출력은 실패로 처리한다.

고정 GT 사본의 이미지와 기존 draw30 provenance의 064 ID 교집합을 정렬해 처음 20장을 선택했습니다. 전체 ID·행별 수치는 `weights/20261026_weatherfar/verification_frames.json`, `weights/20261026_weatherfar/verify.json`에 있습니다. 정본 검증은 원본 draw30_gt.py의 build/infer와 ab_base.segmentation_forward 함수를 AST로 그대로 실행하고 final_pack 원본 함수를 render 순서로 호출하여 비교했습니다.

- PyTorch ↔ 기존 경로: **PASS**, 20장 클래스·개수 동일 및 박스 차 <1px, 최소 마스크 일치율 **100.000000%**.
- (b) ONNX Runtime CPU ↔ 엄격 FP32 PyTorch: **PASS**, 20/20장 클래스·개수 동일, 최대 박스 차 **0.002655px**, 최소 마스크 일치 **100%**.
- (a) ONNX Runtime CPU ↔ 배포 AMP FP16 PyTorch: **PASS**, 20/20장 클래스·개수 동일, 최대 박스 차 **2.909180px**, 최소 마스크 일치 **99.970245%**.
- 원출력 최대 절대차 (ONNX FP32 ↔ 채택 AMP): `{"seg_logits": 0.09027099609375, "det_logits": 0.29559898376464844, "det_distances": 11.774238586425781, "det_priors": 0.0}`.
- 원출력 최대 절대차 (ONNX FP32 ↔ 엄격 PyTorch FP32, TF32 끔): `{"seg_logits": 5.14984130859375e-05, "det_logits": 0.000179290771484375, "det_distances": 0.007747650146484375, "det_priors": 0.0}`.
- batch=1 torch 참고 속도: 평균 **25.41 ms/장**, 중앙값 21.52, p95 36.83. resize/H2D/AMP 두 모델/검출 NMS/D2H 포함, 최종 원본 해상도 NumPy 후처리는 제외, 워밍업 3회 후 CUDA 동기화.

FP32 ONNX와 채택 FP16 AMP는 다른 정밀도입니다. 현재 승인은 TF32를 끈 FP32 비교(b)로 결정하며, 이전 AMP 박스 <1px 기준의 18/20장 결과를 대체합니다. 임계값·NMS·박스 반올림 경계는 작은 원출력 오차를 결과 차이로 증폭할 수 있습니다. 행별 수치를 확인하십시오. TensorRT FP16은 AGX에서 별도로 동치를 검증해야 합니다.

검증 재실행 예(지정 버전의 `verification_frames.json` 순서를 사용하며 `--frames`로 목록을 명시할 수도 있음):

```bash
python tools/verify_parity.py --version 20261026_weatherfar --images /frozen/root/images --provenance /path/provenance.csv --count 20 --reference-source /path/robot_vision3_autolabel --output /path/verify
```

정본 저장소가 없는 AGX에서는 `--reference-source`를 생략해 패키지 내부 torch/ONNX 동치만 확인할 수 있습니다. 자동 검증은 PyTorch 정본 실패 또는 엄격 FP32(b) 미달 시 오류를 반환하며 AMP 참고(a) 미달은 경고입니다.

## 모델 교체 절차

버전은 `weights/YYYYMMDD_이름/` 단위입니다. `seg.pt`, `det.pt`, 두 ONNX,
`model_config.yaml`, `postprocess.yaml`, `verify.json`, `verification_frames.json`,
`SOURCE.txt`를 함께 보관하십시오. `SOURCE.txt`에는 원본 경로·MD5·채택일이 있습니다.
추론기 인스턴스는 생성 시 active_model과 해당 후처리 값을 한 번 읽습니다. 변경 후 서버를 재시작하십시오.

1. 서버에서 `PYTHON=/path/to/python tools/update_from_checkpoint.sh 이름 /path/seg.pt /path/det.pt` 실행.
   서버 기본 Python은 `/home/hs/rang/X-AnyLabeling-Server/.venv/bin/python`입니다.
   오늘 날짜의 새 폴더를 만들고 복사→ONNX export/checker→고정 20장 검증을 수행합니다.
   기존 이름은 덮어쓰지 않습니다. config.yaml은 자동 변경하지 않습니다.
2. `verify.json`의 `b.passed: true`와 `onnx_target_pass: true`, 20장 결과 및 SOURCE를 확인합니다. `a`는 참고 결과입니다.
   통과 기준은 기존 pack과 동일하게 클래스·개수 일치, 박스 좌표 차 **<1px**, 마스크 일치 **≥99.5%**입니다.
   원본 채택 PyTorch 참조 통과는 동일 체크포인트·후처리 복제에서만 승계됩니다.
   새 학습 체크포인트는 strict state 로드와 ONNX↔PyTorch 검증을 수행하며 외부 학습 추론 경로는 실행하지 않습니다.
   실패는 비영 종료와 버전의 FAILED/INCOMPLETE로 표시되며 활성화하면 안 됩니다.
3. 검증 통과한 **버전 폴더 전체**를 AGX 패키지의 `weights/`로 복사합니다.
   이 버전 구조를 지원하는 패키지 코드도 함께 배포하십시오.
4. AGX에서 `PYTHON=python3 bash tools/build_trt.sh /path/vision_model YYYYMMDD_이름` 실행.
   활성 버전 변경 전에 새 버전의 엔진을 생성합니다. TensorRT 정확도는 AGX에서 별도 검증합니다.
5. AGX `config.yaml`의 `active_model: YYYYMMDD_이름`만 변경하고 추론 프로세스를 재시작합니다.
   이전 버전 폴더는 보존하십시오.
6. 되돌리기: `PYTHON=python3 tools/rollback.sh YYYYMMDD_이전이름` 실행 후 프로세스 재시작.
   폴더·20장 검증 통과·검증 당시 파일 SHA256를 확인하고 **active_model 한 줄만** 원자적으로 바꿉니다.

후처리 값은 final_pack.py의 상수 AST를 한 번 읽어 저장합니다. 지원하는 상수 이름·타입,
클래스 키, box bias 키, 범위가 바뀌면 중단합니다. 지원하지 않는 구조는 패키지 구현과
`tools/postprocess_schema.json`을 검토해 함께 수정해야 합니다. final_pack 함수는 import하지 않습니다.

현재 `20261026_weatherfar`는 기존 채택 버전으로 유지했지만 ONNX 목표 미달 이력이 있습니다.
따라서 이 버전의 복제도 동일 기준에서 업데이트가 실패하며, rollback 도구도 이 버전을
검증 통과 버전으로 허용하지 않습니다. 기준을 완화하거나 실패 보고서를 PASS로 바꾸지 마십시오.

## 실시간 실행

`python -m vision_model.runtime --config config/runtime.yaml` 또는
`PYTHON=python3 tools/run_camera.sh --source v4l2 --uri /dev/video0`으로 실행합니다.
`--backend torch|onnx|tensorrt`, `--max-frames 200`으로 덮어쓸 수 있습니다.
설정은 시작할 때 한 번 읽으며 변경 후 재시작합니다. 로봇 구동 연결은 포함하지 않습니다.
아래 YAML은 `source` 항목에 적용하며 나머지 기본 설정은 유지합니다.

USB/UVC:
```yaml
source:
  type: v4l2
  uri: /dev/video0
  width: 1280  # height: 720, fps: 30
```
CSI (GStreamer·nvarguscamerasrc 지원 OpenCV 필요):
```yaml
source:
  type: csi
  csi_sensor: 0
  width: 1280  # height: 720, fps: 30
```
RTSP:
```yaml
source:
  type: rtsp
  uri: rtsp://camera-host/stream
  timeout_s: 3.0
```
ROS2 (`rclpy`, `sensor_msgs`, `std_msgs` 및 ROS 환경 필요):
```yaml
source:
  type: ros2
  topic: /camera/image_raw
  latest_only: true
```

`source.rotate: 0|90|180|270` 또는 `--rotate`로 추론 전 시계 방향 회전을 지정합니다.
`source.speed: 1.0` 또는 `--speed 0.25~4.0`으로 파일·디렉터리 재생 속도를 설정합니다.
파일은 영상 FPS, 디렉터리는 `source.fps` 기준으로 재생하며 빠른 배속에서는 프레임을 건너뜁니다.
카메라·RTSP·ROS2 입력은 배속을 무시하고 경고합니다.
화면에서 `r`은 다음 프레임부터 90도 회전하고 `+`/`-` 또는 `]`/`[`는 배속 단계를 바꿉니다.
`space`는 파일·디렉터리를 일시정지/재개하며 `q` 또는 ESC는 종료합니다.
화면 창은 마우스로 크기를 조절하며 `sinks.display.width` 또는 `--window-width`로 시작 폭을 지정합니다(기본 1280, 비율 유지).
`f`는 전체화면을 전환하고 `1`은 시작 크기로 복원합니다.
파일·디렉터리에서 좌우 화살표는 5초, 상하 화살표는 30초 이동하며 `sinks.display.seek_seconds`로 변경할 수 있습니다. 일시정지 중에도 이동합니다.
화면 좌상단에 배속·회전 각도·처리 FPS와 창 키 안내가 표시됩니다.

`file`은 `uri`에 동영상, `dir`은 프레임 폴더를 지정합니다. 디렉터리는 이름순입니다.
실시간 기본값 `latest_only: true`는 한 칸 버퍼를 덮어써 지연 누적을 막습니다.
유한 입력을 누락 없이 시험할 때는 `latest_only: false`, `source.limit: 200`(dir),
`monitor.max_frames: 200`을 설정합니다. 정상 EOF는 종료하며 읽기 오류는 백오프로 재연결합니다.
파일 재연결은 처음부터, 디렉터리 읽기 오류는 실패 위치부터 재시도합니다.

TensorRT는 대상 AGX에서 `bash tools/build_trt.sh`로 생성한 활성 버전의
`seg.fp16.engine`, `det.fp16.engine`을 사용합니다. TensorRT **10** Python 및
`cuda-python`의 `cuda.bindings.runtime` 또는 `cuda.cudart`가 필요합니다.
이름 기반 `set_tensor_address`·`execute_async_v3`는
[NVIDIA 공식 마이그레이션 문서](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration-guide.html)의 8→10 API를 따릅니다.
기존 torch/ONNX 원출력과 `preprocess.finish`를 그대로 사용하고 TensorRT도 같은 디코드·후처리를 적용합니다.
TensorRT 정확도와 하드웨어 입력은 AGX에서 검증해야 합니다.

`decision`의 ROI는 화면 비율 좌표이며 위험 비율의 분모는 ROI 픽셀 수입니다.
위험 박스는 ROI와 픽셀 교집합이 있으면 반영합니다. 장애물 아래변이
`1-obstacle_bottom_margin`보다 아래면 STOP입니다. 나머지 위험은 SLOW입니다.
**모든 판단 기본값은 임시값, 현장 조정 필요**입니다. 시작 상태는 STOP,
STOP은 즉시 적용하고 회복·기타 상태 변경에는 `stable_frames` 연속 판정이 필요합니다.
입력 타임아웃·읽기 오류·오래된 프레임·추론 오류에는 STOP JSON을 출력합니다.
추론 오류는 STOP 출력 후 비영 종료하며 systemd 예시가 재시작하도록 구성했습니다.

`sinks`에서 json·udp·video·display·ros2를 동시에 켤 수 있습니다.
JSONL에는 UNIX 시각, 입력 프레임 번호, 출력 순번, 판정·유지 이유·후보 판정,
박스·클래스별 ROI 비율, 입력/추론/후처리/판단 ms 및 이동평균 FPS가 기록됩니다.
FPS는 출력·렌더링 비용이 포함된 프레임 처리 간격 기준이며, 입력 ms는 읽기 및
오프라인 생산자 대기, frame_age_ms는 버퍼 체류 시간입니다. 추론 ms에는 기존 predictor의
전처리·디코드가 포함됩니다. video는 기존 2패널 그림에 판정을 올린 mp4입니다.
오류 이벤트는 이미지가 없으므로 JSON/UDP/ROS 판단으로 전달되며 영상에는 쓰지 않습니다.
UDP는 한 이벤트당 JSON 패킷 하나(최대 65507 bytes), ROS 판단은 String JSON,
박스는 String JSON, 마스크는 mono8 Image(0/1/2)입니다. UDP 수신 보장은 없습니다.
화면 서버가 없으면 display를 자동으로 끕니다. SIGINT/SIGTERM 시 STOP을 출력하고 정리합니다.
`config/vision-model.service`는 수동 검토·경로 수정용 예시이며 설치하지 않았습니다.

## ROS2 Jazzy 비전 연결 (기존 runtime 확장)

ROS2 패키지를 새로 만들지 않고 기존 `vision_input` 구독 노드와 `vision_output`
발행 노드를 같은 런타임에서 사용합니다. 기존 `Backend`가 활성 모델을 한 번
로드하며 `Source → Backend → Stabilizer → Decision → Sinks` 순서로 처리합니다.
ViT 분류기는 이 배포 코드에 없으며 연결하지 않습니다. 학습·UI는 변경하지 않습니다.

실행 전 ROS2 Jazzy와 해당 Python의 `rclpy`, `sensor_msgs`, `std_msgs`, 기존 모델
의존성이 필요합니다. Jazzy 설치만으로 AGX의 JetPack/Python/torch 조합이 호환된다고
가정하지 마십시오. 이 변경으로 새 pip 의존성이나 사용자 메시지 패키지는 추가하지 않습니다.

기존 운영 설정을 복사해 ROS 입력·출력만 켭니다. 실제 카메라 토픽이 다르면 복사한
설정의 `source.topic`을 로봇팀 설정에 맞춥니다. 기본 카메라 토픽은 기존 설정값입니다.

```bash
source /opt/ros/jazzy/setup.bash
# 기존 모델 의존성이 설치된 Python 환경을 활성화한 뒤 실행
python - <<'PY'
from pathlib import Path
import yaml
c = yaml.safe_load(Path('config/runtime.yaml').read_text())
c['source']['type'] = 'ros2'
c['sinks']['ros2']['enabled'] = True
# CPU 검증 환경에서만 c['backend']['device'] = 'cpu'로 지정
Path('outputs/runtime').mkdir(parents=True, exist_ok=True)
Path('outputs/runtime/ros2.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
python -m vision_model.runtime --config outputs/runtime/ros2.yaml
```

다른 ROS2 노드와 동일한 ROS domain 및 시계를 사용해야 합니다. rosbag의 simulated
clock을 사용하면 두 노드에 `--ros-args -p use_sim_time:=true`를 적용하고 `/clock`을
공급하십시오. stamp 0·미래 stamp·`source.timeout_s`보다 오래된 이미지는 오류로
처리합니다. 영상 회전이 필요하면 ROI와 픽셀 박스 좌표계도 함께 검증하십시오.

| 방향 | 기본 토픽 | 메시지 / QoS | 계약 |
|---|---|---|---|
| 구독 | `/camera/image_raw` | `sensor_msgs/msg/Image`, BEST_EFFORT·VOLATILE·KEEP_LAST(1) | bgr8/rgb8/mono8/bgra8/rgba8, stride padding 지원 |
| 발행 | `/vision/mask` | `sensor_msgs/msg/Image`, RELIABLE·VOLATILE·depth 1 | 원본 stamp/frame_id, mono8, 0=drivable/1=caution/2=non_drivable |
| 발행 | `/vision/boxes` | `std_msgs/msg/String`, RELIABLE·VOLATILE·depth 10 | 기존 JSON boxes 계약 유지, header/추론 상태/클래스명 추가 |
| 발행 | `/vision/decision` | `std_msgs/msg/String`, RELIABLE·VOLATILE·depth 10 | 기존 JSON decision 계약 유지, 아래 메타데이터 추가 |

JSON 문자열은 기존 실행 코드의 인터페이스이므로 유지합니다. 신규 인터페이스가 없어
JSON을 새로 선택한 것이 아닙니다. `boxes`는 기존 시간축 안정화가 적용된
`[[x1,y1,x2,y2], score, class_id]` 목록이며 원본 이미지 픽셀 좌표입니다.
검출 클래스 0=step/1=ditch_hole/2=puddle/3=obstacle입니다. 트랙 확인·유지 때문에
단일 프레임 모델 검출과 발행 박스가 달라질 수 있습니다.

기존 decision 응답 키는 유지하며 `header`(정수 sec/nanosec, frame_id,
수신 시 age_at_receive_ms), `inference_status`, `decision_valid`,
`policy_validated`, `detected_classes`, `last_success_timestamp`를 추가합니다.
`timestamp`는 ROS 입력에서 촬영 시각이며 정확한 나노초는 `header.stamp`를 사용합니다.
입력 오류처럼 대응 프레임이 없는 이벤트의 header는 null입니다.
`frame_age_ms`는 수신 전 지연과 로컬 대기·추론 시간을 합친 값입니다.
`latency_ms`·`fps`는 기존 이동평균 계측을 계속 제공합니다.

정책은 기존 `config/runtime.yaml`의 임시값을 그대로 보존합니다.
ROI는 [(0.35,0.55),(0.65,0.55),(0.95,1),(0.05,1)], ROI 내 non_drivable
비율 >0.30은 STOP 후보, caution >0.20은 SLOW 후보입니다. ROI와 겹친 obstacle은
박스 하단이 영상 높이의 0.80 이상일 때 STOP 후보, 그 외 SLOW 후보입니다.
step/ditch_hole/puddle은 ROI 겹침 시 SLOW 후보입니다. 기존 STOP 2프레임,
회복 8프레임, 기타 전환 3프레임과 트랙 안정화 설정도 유지합니다.
이 값은 검증된 안전 기준이 아니며 픽셀 위치로 실제 거리·깊이·턱 높이를 측정하지 않습니다.

`decision.policy_validated: false`가 기본입니다. ROS 입력 또는 ROS 출력 사용 시
정책 검증 전에는 항상 STOP, reasons=[policy_unvalidated], decision_valid=false를
발행하고 candidate/candidate_reasons/roi_ratios로 기존 정책의 후보 근거를 제공합니다.
로봇팀이 카메라 장착·주행 폭·ROI·클래스 위험 수준·연속 프레임 수를 검증한 뒤에만
해당 설정을 true로 바꿔야 합니다. 검출 신뢰도 임계값은 활성 모델의 postprocess.yaml을
그대로 사용하며, 후보 GO는 현재 비전에서 제한 근거를 찾지 못했다는 의미입니다.

ROS 경로는 실행 중인 추론 1개와 최신 대기 프레임 1개만 유지하며 추가 선행 추론을
하지 않습니다. 기존 `source.timeout_s`(기본 3초)를 입력 중단·프레임 최대 지연
감시에 재사용했습니다. 이 시간도 로봇팀의 제동/감시 요구에 맞춰 검증해야 합니다.
입력 오류·추론 오류·지연·종료는 즉시 STOP 권고와 오류/종료 상태를 발행하고
시간축 안정화 이력을 초기화합니다. 지연된 추론 결과는 폐기합니다.
진행 중인 GPU/네이티브 연산을 강제 종료하지 않으므로 종료 대기가 생길 수 있습니다.
노드 자체의 종료·통신 단절에 대비한 최종 수신 유효시간 판단과 모터 제어는 로봇팀
책임입니다. 비전은 `/cmd_vel`을 발행하지 않습니다.

```bash
ros2 node list
ros2 topic list
ros2 topic info /camera/image_raw --verbose
ros2 topic info /vision/decision --verbose
ros2 topic echo /vision/decision --once --full-length
ros2 topic echo /vision/boxes --once --full-length
ros2 topic echo /vision/mask --field header --once
python tools/verify_ros2_runtime.py
```

기본 회귀 검사는 DDS/모델 없이 이미지 변환·stamp·정책 잠금·입력 중단·추론 오류·
지연 폐기를 확인합니다. 실제 장비 검증에서는 카메라 중단 후 STOP/error,
모델 오류 후 GO 미발행, 기존 추론기와 동일 이미지의 mask/boxes 동치도 확인해야 합니다.
EOF나 max_frames 종료도 ROS 출력에는 STOP/shutdown으로 전달합니다.

## 기존 영상으로 ROS2 후보/최종 판정 확인

`python -m vision_model.runtime.ros2_video publish`는 기존 `Source`의 파일 읽기,
원본 FPS 기반 재생·최신 프레임 버퍼·EOF 처리를 재사용합니다. 모델을 생성하지 않고
디코딩된 BGR 원본을 `source.topic`에 발행합니다. 해상도를 축소하거나 draw 화면을
입력으로 보내지 않습니다. publisher의 stamp는 재생 시점의 ROS clock이며, 선택
JSONL에는 원본 영상 프레임 번호와 영상 시간이 별도로 기록됩니다.

모델은 기존 비전 runtime에서만 로딩합니다. 기존 `candidate`는 안정화된 검출로
계산한 공간 정책 후보, `decision`은 정책 보호가 적용된 최종 발행 상태입니다.
이름을 바꾸거나 같은 의미의 candidate_decision/published_decision을 추가하지 않습니다.
기존 필드로 알 수 없었던 검출 안정화 전 비교만 `pre_stabilizer`에 추가했습니다.
이 필드의 candidate/candidate_reasons는 동일한 `Decision.evaluate` 공간 정책으로
계산하며, 추가 모델 추론이나 별도 시간축 상태를 만들지 않습니다. 검출 수와 클래스도
안정화 전후로 비교할 수 있습니다. 마스크는 기존 검출 후처리·box bias 결과를 그대로
공유하므로 이 비교는 검출 트랙 안정화의 영향이며 세그 후처리를 제거한 비교는 아닙니다.

`watch`는 실제 `/vision/decision` 구독 결과를 사람이 읽기 쉽게 출력하고, `--output`을
지정하면 수신 JSON 원문과 요약을 남깁니다. Frame은 runtime 수신 번호이며 원본 영상
번호와 다를 수 있습니다. 정확한 원본 번호는 발행 JSONL의 stamp와 결과 header.stamp를
연결해 확인합니다. 단순 `ros2 topic echo`에서도 candidate/decision을 모두 볼 수 있습니다.

기존 `draw/draw_gt.py`는 정지 이미지와 대응 GT 라벨을 그리는 별도 도구입니다.
이번 영상 발행기는 이를 실행하지 않습니다. 기존 `vision_model/draw.py`의 예측 오버레이는
비전 runtime의 display/video sink가 이미 추론한 결과로 그립니다. 따라서 GT 렌더러,
GUI 재생, 기존 저장 옵션을 보존하면서 이 테스트에서 모델이 두 번 실행되지 않습니다.
MP4에 대응하는 GT가 없으면 GT 검증을 수행했다고 간주하지 않습니다.

터미널마다 프로젝트 루트에서 ROS 환경과 기존 모델 Python 환경을 활성화하십시오.
개발 PC에서 최종 통합 검증한 Torch CUDA 설정은 `outputs/runtime/ros2_video_cuda.yaml`입니다.
ONNX CPU 부분 시험 설정은 `outputs/runtime/ros2_video_cpu.yaml`에 보존했습니다.
활성 버전의 기존 seg.onnx/det.onnx를 사용하며 Torch/CUDA 운영 기본 설정은 변경하지 않습니다.
AGX에서는 `outputs/runtime/ros2.yaml`처럼 기존 CUDA 설정을 사용하며 실기 검증이 필요합니다.
아래 스레드 환경변수는 테스트 프로세스의 CPU 병렬 처리를 제한하는 실행 설정입니다.
이 환경에서 PyTorch CPU는 3초 예산을 반복 초과했으며, 스레드 제한만으로 해결되지 않았습니다.
ONNX CPU 시험은 28프레임을 처리했고, 최종 30프레임 시험은 개발 PC RTX 4070 Ti에서 수행했습니다.
첫 실행의 정상 결과 6프레임과 후속 지연 실패 기록은 attempt1/attempt2 산출물로 보존했습니다.
정책값·모델 수치 연산 순서·원본 FPS를 바꾸는 옵션은 아닙니다.

터미널 1 — 비전 노드:
```bash
source /opt/ros/jazzy/setup.bash
source /home/hs/rang/X-AnyLabeling-Server/.venv/bin/activate
export OPENCV_FOR_THREADS_NUM=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m vision_model.runtime --config outputs/runtime/ros2_video_cuda.yaml
```
이 CUDA 검증 설정은 30개 정상 프레임 후 종료하고 예측 MP4를 저장합니다.
계속 실행하려면 `--max-frames 0`을 붙입니다. 안전 정책 플래그는 false로 유지하십시오.

터미널 2 — 영상 발행:
```bash
source /opt/ros/jazzy/setup.bash
source /home/hs/rang/X-AnyLabeling-Server/.venv/bin/activate
export OPENCV_FOR_THREADS_NUM=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m vision_model.runtime.ros2_video publish \
  --config outputs/runtime/ros2_video_cuda.yaml \
  --uri '/home/hs/다운로드/영상/과수원/063_5min.mp4' \
  --frame-id video_camera_optical --wait-for-subscriber 10 \
  --output outputs/runtime/video_publish.jsonl
```
`--uri`와 `--frame-id`는 테스트 입력·카메라 좌표계에 맞게 바꿀 수 있습니다.
영상 파일 끝에서 발행기는 정상 종료합니다. `--max-frames`는 발행 수 제한,
`--speed`는 기존 0.25/0.5/1/2/4배 재생 설정입니다. 기본은 원본 FPS·1배속이며
반복 재생 옵션은 기존 Source에 없어 새로 추가하지 않았습니다.

터미널 3 — 실제 후보와 발행 상태:
```bash
source /opt/ros/jazzy/setup.bash
source /home/hs/rang/X-AnyLabeling-Server/.venv/bin/activate
python -m vision_model.runtime.ros2_video watch \
  --config outputs/runtime/ros2_video_cuda.yaml \
  --output outputs/runtime/video_decisions.jsonl
# 또는 기존 ROS CLI에서 JSON 전체 확인
ros2 topic echo /vision/decision std_msgs/msg/String --full-length
```
감시기는 `--max-frames 30` 또는 `--timeout 120`으로 종료 조건을 줄 수 있습니다.
카메라 발행기와 비전 노드·감시기에 같은 ROS_DOMAIN_ID를 적용하십시오. 정책 보호가
유지된 상태에서 Candidate는 변할 수 있지만 Published는 STOP, Valid는 false입니다.
입력 중단/추론 오류 이벤트는 candidate도 STOP으로 초기화하며 이전 후보를 유지하지 않습니다.
EOF 후 비전 노드를 계속 실행하면 기존 timeout 감시가 STOP/error를 발행합니다.

검증 기록은 `outputs/runtime/ros2_video_test.json`, 실제 수신 JSONL과 터미널 로그에
저장합니다. 발행 FPS·처리 FPS·스킵/드롭은 구별해야 합니다. Source의 dropped는
구독 콜백 이후 최신 버퍼에서 교체한 수이며 DDS에서 콜백 이전에 누락된 프레임이나
시험 종료 뒤 발행된 프레임까지 포함한 전체 손실 수는 아닙니다.
EOF 전용 별도 MP4 검사는 `outputs/runtime/ros2_video_eof_report.json`을 참고하십시오.
최종 보고서의 성능은 개발 PC RTX 4070 Ti CUDA 결과이며, 별도 CPU 기록과 AGX 성능을 구분하십시오.

실제 통합 결과(`063_5min.mp4`, 1920×1080, 원본 10 FPS, 개발 PC RTX 4070 Ti):
발행 118프레임·9.90 FPS, 정상 추론/판정 수신 30프레임·실행 처리량 3.73 FPS,
평균 모델 추론 173.50 ms·프레임 전체 지연 356.17 ms였습니다. 발행기 스킵 2,
런타임 최신 버퍼 교체 21이며 이 두 값은 DDS 손실이나 종료 후 발행 수를 포함하지 않습니다.
후보 GO 1/SLOW 9/STOP 20, 최종 STOP 30·decision_valid=false였습니다.
초기 GO 1회는 검출 트랙 확인 전 박스가 숨겨진 경우였으며, pre_stabilizer 후보는 STOP입니다.
30개의 결과 header는 발행 로그와 일치했고, 마스크 30개를 DDS로 수신했습니다.
모델은 한 번 로딩했으며 동일 predictor의 안정화 전 mask/boxes 동치 검사도 통과했습니다.
기존 video sink로 30프레임 예측 오버레이 MP4를 저장했습니다.

검사 실패도 기록했습니다. CPU 지연 실행 외에, 최종 GPU 시험의 보조 BEST_EFFORT
원본 관찰자가 모든 프레임을 받아야 한다는 assertion은 일부 누락으로 실패했습니다.
완료된 모델 추론을 반복하지 않고, 실제 발행 로그와 실제 수신 판정 30개로 header를
대조했습니다. 별도 원본 3프레임의 수신 BGR 픽셀 일치 검사도 통과했습니다.
boxes 수신 계약은 저장된 실제 검출 결과를 추가 추론 없이 재발행해 확인했습니다.
시작 시 입력 대기 timeout 1회는 STOP/error로 발행됐고, 정상 모델 프레임 30개와 구분했습니다.
EOF 별도 시험은 기존 demo_video.mp4의 249프레임을 모두 발행·수신하고 정상 종료했습니다.
실카메라·AGX·동영상 GT 검증은 수행하지 않았습니다.

재실행 설정의 JSON/MP4 출력 경로는 `ros2_video_live_*`로 지정해 위 검증 증거를
보존합니다. 검증 중인 3개 터미널은 같은 ROS domain을 사용해야 합니다.
EOF와 Ctrl+C는 영상 발행기를 종료하지만, 비전 노드의 입력 timeout 보호는 계속 유지됩니다.

## ViT-S 조언 (ADVICE, 11-18)

버전 폴더 `postprocess.yaml`에 `ADVICE: {thresh: 0.9, scale: 1.1, classes: [0,1,2,3]}`가 있으면, 검출 박스마다 crop 분류기(DINOv2-small ViT-S, crop_verifier 채택 M_all)로 판정합니다. 분류기가 박스와 다른 클래스를 0.9 이상 확신하면 그 박스를 지웁니다. 클래스·점수는 바꾸지 않습니다. 없으면 기존과 같습니다. 런타임 설정 `backend.advice: false`로도 끌 수 있습니다.

- 버전: `weights/20261118_s50vit` (seg·det는 20261113_s50과 같은 파일, ADVICE만 추가)
- 파일: `advice.json`(전처리 상수·출처 md5), `advice.ts`(torch 백엔드, transformers 불필요), `advice.onnx`(onnx 백엔드), `advice.fp16.engine`(tensorrt, AGX에서 `bash tools/build_trt.sh`가 정적 배치 8로 생성)
- 생성: PC에서 `python tools/export_advice.py --checkpoint <M_all.pt> --folder weights/<버전>` (transformers 필요)
- PC 검증: 시험 226장에서 robot3 평가 구현과 남은 박스가 226/226장 같음(torch), onnx 20/20장 같음. 조언 단계 중앙 11ms·p95 16ms(4070 Ti). latency에 `advice` ms로 기록됩니다.
- AGX: TensorRT 엔진의 정확도·속도는 아직 재지 않았습니다. torch 백엔드(`advice.ts`)는 AGX torch로도 실행됩니다.
