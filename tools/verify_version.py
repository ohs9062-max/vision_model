#!/usr/bin/env python3
# 고정 20장으로 새 버전의 FP32 승인 기준과 기준 버전 일치 여부를 검사한다.
"""Fixed 20-frame update gate using the original pack's numerical criteria."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vision_model.infer_torch import TorchPredictor
from vision_model.infer_onnx import OnnxPredictor
from verify_parity import compare, evaluate_parity, summarize_parity

# 파일의 SHA256을 계산한다.
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

# 명령행 인자를 해석하고 검증을 실행한다.
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--version',required=True); parser.add_argument('--baseline',required=True)
    parser.add_argument('--smoke',action='store_true'); a=parser.parse_args()
    package=Path(__file__).resolve().parents[1]; torch.set_num_threads(4)
    # 1) 대상·기준 모델과 고정 입력 불러오기
    current=TorchPredictor(package,version=a.version)
    baseline=TorchPredictor(package,version=a.baseline)
    onnx=OnnxPredictor(package,version=a.version)
    target=current.version_dir
    frames=json.loads((target/'verification_frames.json').read_text())
    if len(frames)!=20 or len(set(frames))!=20: raise ValueError('Need exactly 20 frozen frames')
    identical=all(sha(target/name)==sha(baseline.version_dir/name) for name in ('seg.pt','det.pt','postprocess.yaml','model_config.yaml'))
    previous=json.loads((baseline.version_dir/'verify.json').read_text())
    rows=[]
    # 2) 고정 입력마다 정밀도 비교와 복제 일치 검사하기
    for i,path in enumerate(frames[:1] if a.smoke else frames,1):
        start=time.perf_counter(); image=cv2.imread(path)
        if image is None: raise FileNotFoundError(path)
        actual=current(image); expected=baseline(image); exported=onnx(image)
        parity=evaluate_parity(current,onnx,image,actual)
        record={'frame':Path(path).stem,'baseline_torch':compare(expected,actual),'onnx_torch':parity['a'],**parity}
        if identical:
            record['baseline_exact']=(expected['boxes']==actual['boxes'] and np.array_equal(expected['mask'],actual['mask']))
            if not record['baseline_exact']: raise RuntimeError('Identical checkpoint copy changed output')
        rows.append(record)
        stamp=subprocess.check_output(['date','+%F %T'],text=True).strip()
        line=f'{stamp} 처리 {i}/{1 if a.smoke else 20}장 {time.perf_counter()-start:.3f} s/장\n'
        with (target/'STATUS').open('a') as stream: stream.write(line)
        if os.environ.get('TH_JOB_TASK'):
            with (Path(os.environ['TH_JOB_TASK'])/'STATUS').open('a') as stream: stream.write(line)
        print(line,end='',flush=True)
        (target/'verify.partial.json').write_text(json.dumps(rows,indent=2))
    if a.smoke:
        print('smoke passed: models load, numeric outputs produced, identical copies match',flush=True); return
    # 3) 엄격 기준 판정과 검증 파일 해시 저장하기
    summaries=summarize_parity(rows); passed=summaries['b']['passed']
    artifacts={name:sha(target/name) for name in ('seg.pt','det.pt','seg.onnx','det.onnx','postprocess.yaml','model_config.yaml','verification_frames.json')}
    report={**summaries,'approval_basis':'b','frames':20,'criteria':summaries['b']['criteria'],'onnx_target_pass':passed,'torch_reference_pass':previous.get('torch_reference_pass') if identical else None,'torch_reference_note':'Original pack reference PASS inherited only for byte-identical checkpoints, model config and postprocess; baseline outputs checked exactly' if identical else 'New checkpoints: architecture/state validated by strict load; original external training pipeline not executed','baseline_version':a.baseline,'identical_bundle':identical,'baseline_exact_pass':all(r.get('baseline_exact',False) for r in rows) if identical else None,'onnx_min_mask':min(r['onnx_torch']['mask_agreement'] for r in rows),'artifact_sha256':artifacts,'checkpoint_sha256':{k:artifacts[k+'.pt'] for k in ('seg','det')},'rows':rows}
    (target/'verify.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','artifact_sha256')},indent=2),flush=True)
    if not passed: raise RuntimeError('Strict FP32 ONNX parity (b) failed: class/count equality, box <=1px, mask >=99.9% required')

if __name__=='__main__': main()
