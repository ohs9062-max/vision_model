#!/usr/bin/env python3
import argparse, json, os, sys, time
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from vision_model.runtime.__main__ import load_config
from vision_model.runtime.backends import Backend
from vision_model.runtime.decision import Decision
from vision_model.draw import draw

ROOT = Path(__file__).resolve().parents[1]
STATUS = Path('/home/hs/rang/robot_vision3_autolabel/outputs/runs/vision_video_export_20261027/STATUS')
def status(s):
    with STATUS.open('a') as f: f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + s + '\n')
def count_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise OSError('cannot open '+str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release(); return n

def run(path, out, backend, config, rotate, limit=0):
    path = Path(path); out.mkdir(parents=True, exist_ok=True)
    video = out / (path.stem + '_pred.mp4'); js = out / (path.stem + '_pred.jsonl')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise OSError('cannot open '+str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or fps <= 0: raise ValueError('invalid input metadata')
    if limit: total = min(total, limit)
    if not limit and video.exists() and js.exists() and count_video(video) == total:
        with js.open() as f: lines = sum(1 for _ in f)
        if lines == total:
            status(f'skip {path.name} {total} frames'); cap.release(); return
    if rotate is not None: cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    effective_rotate = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) if rotate is None else rotate
    if effective_rotate not in (0,90,180,270): raise ValueError('unsupported rotation')
    decision = Decision(config['decision']); history = deque(maxlen=config['monitor']['window']); ticks = deque(maxlen=config['monitor']['window'])
    writer = None; start = time.perf_counter(); n = 0
    try:
        with js.open('w') as f:
            while n < total:
                t = time.perf_counter(); ok, image = cap.read(); input_ms = (time.perf_counter()-t)*1000
                if not ok: raise EOFError(f'input ended at {n}/{total}')
                if rotate: image = cv2.rotate(image, {90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate])
                result, stages = backend(image)
                if not np.isfinite(result['mask']).all() or any(not np.isfinite(b).all() or not np.isfinite(s) for b,s,c in result['boxes']): raise FloatingPointError('nonfinite predictor result')
                t = time.perf_counter(); state = decision(result); stages.update(input=input_ms, decision=(time.perf_counter()-t)*1000); stages['total'] = sum(stages.values())
                history.append(stages.copy()); ticks.append(time.perf_counter()); rate = (len(ticks)-1)/(ticks[-1]-ticks[0]) if len(ticks)>1 else 0.
                n += 1
                packet = dict(timestamp=time.time(), frame=n, event='frame', sequence=n, **state, boxes=result['boxes'], latency_ms=stages, moving_latency_ms={k:float(np.mean([x[k] for x in history])) for k in stages}, frame_age_ms=0, fps=rate, dropped_frames=0, rotate=effective_rotate, speed=1.0, source_frame=n, source_time_s=(n-1)/fps)
                f.write(json.dumps(packet, ensure_ascii=False, allow_nan=False)+'\n')
                panel=draw(image,result)
                cv2.putText(panel, state['decision']+' '+','.join(state['reasons']), (15,panel.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                cv2.putText(panel, f'1x  {effective_rotate} deg  {rate:.1f} FPS', (15,25), cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                size=(1280,max(2,round(panel.shape[0]*1280/panel.shape[1])//2*2)); panel=cv2.resize(panel,size,interpolation=cv2.INTER_AREA)
                if writer is None:
                    for codec in ('avc1','mp4v'):
                        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*codec),fps,size)
                        if writer.isOpened(): break
                        writer.release()
                    else: raise OSError('cannot open video writer')
                writer.write(panel)
                if n%1000==0: status(f'처리 {n}/{total}장 {(time.perf_counter()-start)/n:.3f} s/장 {path.name}')
    finally:
        cap.release()
        if writer: writer.release()
    with js.open() as f: lines=sum(1 for _ in f)
    written=count_video(video)
    if n != total or written != total or lines != total: raise RuntimeError(f'count mismatch input={total} processed={n} video={written} jsonl={lines}')
    status(f'verified {path.name} input={total} output={written} jsonl={lines} seconds={time.perf_counter()-start:.1f} video_bytes={video.stat().st_size} jsonl_bytes={js.stat().st_size}')
    print(f'{path.name} {n} frames {time.perf_counter()-start:.1f}s',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('videos',nargs='+'); p.add_argument('--output',required=True); p.add_argument('--rotate',type=int,choices=(0,90,180,270)); a=p.parse_args()
    config=load_config(ROOT/'config/runtime.yaml'); config['backend'].update(type='torch',device='cuda')
    backend=Backend(config['backend'])
    try:
        for path in a.videos: run(path,Path(a.output),backend,config,a.rotate,int(os.environ.get('VISION_EXPORT_SMOKE_FRAMES','0')))
    finally: backend.close()
if __name__=='__main__': main()
