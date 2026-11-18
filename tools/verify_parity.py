#!/usr/bin/env python3
# 같은 입력의 ONNX 출력을 엄격 FP32 승인(b)과 AMP 참고(a)로 나누어 검증한다.
"""20-frame parity/latency report. --reference-source enables original draw30_gt path."""
import argparse,ast,csv,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vision_model.infer_torch import TorchPredictor
from vision_model.preprocess import finish
from vision_model.postprocess import drop_box_on_drivable

# 현재 시각과 진행 단계를 STATUS에 추가한다.
def stamp(out,message):
    if out:
        with (Path(out)/'STATUS').open('a') as f: f.write(subprocess.check_output(['date','+%F %T'],text=True).strip()+' '+message+'\n')

# 기존 추론 함수와 모델을 불러와 비교용 예측 함수를 만든다.
def reference(root, version_dir=None):
    root=Path(root); sys.path.insert(0,str(root)+'-final-student'); sys.path.insert(0,str(root/'experiments/student_deploy_20261019/trainer_v2'))
    from models.student import build_pidnet,build_rtmdet
    from cspnext_seg import build_cspnext_seg
    namespace=dict(torch=torch,np=np,cv2=cv2,build_pidnet=build_pidnet,build_rtmdet=build_rtmdet,build_cspnext_seg=build_cspnext_seg,Path=Path)
    # Execute exact specified functions without draw30's dataset selection/global side effects.
    for file,names in [(root/'experiments/student_deploy_20261019/trainer_v2/ab_base.py',{'segmentation_forward'}),(root/'experiments/draw30_gt_20261026/draw30_gt.py',{'build','infer'})]:
        source=file.read_text(); tree=ast.parse(source); selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(file),'exec'),namespace)
    file=root/'experiments/student_deploy_20261019/final_pack.py'; spec=importlib.util.spec_from_file_location('reference_final_pack',file); fp=importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
    import yaml
    config=yaml.safe_load((Path(version_dir)/'postprocess.yaml').read_text()) if version_dir else None
    checkpoint_dir=Path(version_dir) if version_dir else root/'outputs/checkpoints/student_20261026_weatherfar'
    models=[namespace['build'](checkpoint_dir/f'{k}.pt') for k in ('seg','det')]
    # 기존 경로의 추론과 후처리를 실행한다.
    def predict(image):
        (sm,sc),(dm,dc)=models; probs=namespace['infer'](sm,sc,image); rec=namespace['infer'](dm,dc,image); h,w=image.shape[:2]
        dets=fp.postprocess_record(rec,w,h,dc['input_size'][1],dc['input_size'][0],post_thresh=config['POST_THRESH'] if config else None,group_config=config); base=np.stack([cv2.resize(probs[c],(w,h),interpolation=cv2.INTER_LINEAR) for c in range(3)])
        base_mask=base.argmax(0).astype(np.uint8); dets=fp.drop_far_from_drivable(dets,base_mask)
        if config and 'OBS_KEEP_RING' in config: dets=fp.keep_obstacle_touching_drivable(dets,base_mask,config['OBS_KEEP_RING'],config['OBS_KEEP_MIN_FRAC'],config['OBS_KEEP_CLASSES'])
        dets=drop_box_on_drivable(dets,base_mask,config.get('DRIVABLE_BOX_DROP') if config else None)
        bias=config['BOX_BIAS'] if config else fp.BOX_BIAS
        mask=fp.apply_box_bias(base,dets,b_obs=bias['obstacle_non'],b_cau=bias['hazard_caution'],min_score=bias['min_score'],obstacle_min_score=config['POST_THRESH'][3] if config and 'OBS_KEEP_RING' in config else None)
        return {'boxes':dets,'mask':mask}
    return predict

# 클래스별 박스를 대응시켜 좌표 차이와 마스크 일치율을 구한다.
def compare(a,b):
    aa=a['boxes']; bb=b['boxes']; remaining=list(range(len(bb))); errors=[]; classes=True
    for box,score,c in aa:
        choices=[j for j in remaining if bb[j][2]==c]
        if not choices: classes=False; break
        j=min(choices,key=lambda j:np.max(np.abs(np.array(box)-bb[j][0]))); remaining.remove(j); errors.append(float(np.max(np.abs(np.array(box)-bb[j][0]))))
    same=len(aa)==len(bb) and classes and not remaining
    return {'boxes_class_count_equal':same,'box_max_abs_px':max(errors,default=0.) if same else None,'boxes_within_1px':same and max(errors,default=0.)<=1.,'mask_agreement':float(np.mean(a['mask']==b['mask'])),'box_counts':[len(aa),len(bb)]}

# 지정 정밀도로 세그멘테이션·검출 원출력을 수집한다.
@torch.inference_mode()
def network_outputs(p,image,amp=True):
    with torch.autocast(p.device.type,enabled=amp):
        seg=p.models['seg'](p.tensor(image,'seg')); m=p.models['det']; det=m.head(m.neck(m.backbone(p.tensor(image,'det'))))
    return [seg.float().cpu().numpy()]+[v.float().cpu().numpy() for v in det]

# autocast와 TF32를 끈 FP32 원출력과 최종 예측을 계산한다.
@torch.inference_mode()
def strict_fp32(p,image):
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.backends.cudnn.flags(allow_tf32=False), torch.autocast(p.device.type,enabled=False):
            seg=p.models['seg'](p.tensor(image,'seg'))
            m=p.models['det']; x=p.tensor(image,'det')
            det=m.head(m.neck(m.backbone(x)))
            rec=m(x)[0]
            probs=seg.float().softmax(1)[0].cpu().numpy()
            arrays=[seg.float().cpu().numpy()]+[v.float().cpu().numpy() for v in det]
        result=finish(probs,{k:v.cpu().numpy() for k,v in rec.items()},image.shape,p.config['det']['input_size'],p.postprocessor)
        return result,arrays
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous


# AMP 참고(a)와 FP32 승인(b)을 같은 ONNX 출력에 비교한다.
def evaluate_parity(p,onnx,image,actual=None):
    actual=p(image) if actual is None else actual
    exported=onnx(image); raw=onnx.outputs(image)
    ort_arrays=[raw['seg'][0]]+raw['det']
    strict,fp32_arrays=strict_fp32(p,image)
    amp_arrays=network_outputs(p,image,amp=p.device.type=='cuda')
    result={}
    # 후처리 오차와 원출력 차이를 기준별로 기록한다.
    for name,expected,arrays,box_limit,mask_limit in [('a',actual,amp_arrays,3.,.995),('b',strict,fp32_arrays,1.,.999)]:
        comparison=compare(expected,exported)
        differences={k:float(np.max(np.abs(x-y))) for k,x,y in zip(['seg_logits','det_logits','det_distances','det_priors'],arrays,ort_arrays)}
        finite=all(np.isfinite(v).all() for v in arrays+ort_arrays)
        passed=finite and comparison['boxes_class_count_equal'] and comparison['box_max_abs_px']<=box_limit and comparison['mask_agreement']>=mask_limit
        result[name]={**comparison,'raw_max_abs':differences,'raw_finite':bool(finite),'passed':bool(passed)}
    return result


# 행별 결과를 기준별로 집계하고 참고 기준 미달을 경고한다.
def summarize_parity(rows):
    summaries={}
    for name,precision,box_limit,mask_limit in [('a','PyTorch AMP FP16 vs ONNX FP32',3.,.995),('b','PyTorch FP32 (autocast/TF32 off) vs ONNX FP32',1.,.999)]:
        records=[r[name] for r in rows]
        summaries[name]={'reference':precision,'approval_gate':name=='b','criteria':{'class_count_equal':True,'box_max_abs_px_inclusive':box_limit,'mask_agreement_min':mask_limit,'raw_max_abs_tolerance':None,'raw_max_abs_note':'Measured, no numeric acceptance tolerance specified; nonfinite outputs fail'},'passed':all(r['passed'] for r in records),'class_count_equal':all(r['boxes_class_count_equal'] for r in records),'box_max_abs_px':max((r['box_max_abs_px'] for r in records if r['box_max_abs_px'] is not None),default=None),'mask_agreement_min':min(r['mask_agreement'] for r in records),'raw_max_abs':{k:max(r['raw_max_abs'][k] for r in records) for k in records[0]['raw_max_abs']}}
    if not summaries['a']['passed']: print('WARNING: AMP FP16 reference (a) missed; approval uses strict FP32 (b)',flush=True)
    return summaries


# 명령행 인자를 해석하고 검증을 실행한다.
def main(a):
    torch.set_num_threads(4); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); root=Path(a.images)
    # 1) 비교할 고정 이미지 선택하기
    manifest=Path(a.frames) if a.frames else (Path(a.package)/'weights'/a.version/'verification_frames.json' if a.version else None)
    paths=[Path(p) for p in json.loads(manifest.read_text())] if manifest else sorted(root.glob('*.jpg'))
    if manifest and any(p.parent.resolve()!=root.resolve() for p in paths): raise ValueError('Frozen frame manifest does not match --images directory')
    if a.provenance:
        ids={r['frame_id'] for r in csv.DictReader(open(a.provenance))}; paths=[p for p in paths if p.stem in ids]
    paths=paths[:a.count]
    if len(paths)!=a.count: raise ValueError(f'Need {a.count} eligible frames, got {len(paths)}')
    (out/'frames.json').write_text(json.dumps([str(p) for p in paths],indent=2))
    # 2) 모델을 불러오고 한 장 추론 점검하기
    p=TorchPredictor(a.package,version=a.version); ref=reference(a.reference_source, p.version_dir if a.version_reference else None) if a.reference_source else None
    from vision_model.draw import draw
    first=cv2.imread(str(paths[0])); smoke=p(first); assert smoke['mask'].shape==first.shape[:2]; assert draw(first,smoke).shape==(first.shape[0],first.shape[1]*2,3)
    # Finite logits and raw decoder smoke checked during full validation.
    if a.smoke:
        if ref:
            v=compare(ref(first),smoke); assert v['boxes_within_1px'] and v['mask_agreement']>=.999,v
        print('smoke passed',flush=True); return
    from vision_model.infer_onnx import OnnxPredictor
    onnx=OnnxPredictor(a.package,version=a.version); rows=[]; latency=[]
    for _ in range(3): p.raw(first)
    # 3) 각 이미지의 두 정밀도 비교와 지연 시간 기록하기
    for i,path in enumerate(paths,1):
        start=time.perf_counter(); image=cv2.imread(str(path)); expected=ref(image) if ref else None
        torch.cuda.synchronize(); t=time.perf_counter(); probs,rec=p.raw(image); torch.cuda.synchronize(); ms=(time.perf_counter()-t)*1000; latency.append(ms)
        actual=finish(probs,rec,image.shape,p.config['det']['input_size'],p.postprocessor); result=onnx(image)
        parity=evaluate_parity(p,onnx,image,actual)
        row={'frame':path.stem,'torch_reference':compare(expected,actual) if expected else None,'onnx_torch':parity['a'],'torch_raw_ms':ms,'raw_max_abs_amp':parity['a']['raw_max_abs'],'raw_max_abs_fp32':parity['b']['raw_max_abs'],**parity}
        rows.append(row); (out/'verify.partial.json').write_text(json.dumps(rows,indent=2)); stamp(out,f'처리 {i}/{len(paths)}장 {time.perf_counter()-start:.3f} s/장')
    # 4) 기준별 결과 집계하고 엄격 기준으로 판정하기
    reference_pass=all(r['torch_reference']['boxes_within_1px'] and r['torch_reference']['mask_agreement']>=.999 for r in rows) if ref else None
    summaries=summarize_parity(rows); onnx_pass=summaries['b']['passed']
    result={**summaries,'approval_basis':'b','frames':len(rows),'selection':'version verification_frames.json order' if manifest else 'sorted frozen-image/provenance intersection, first N','device':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'torch_reference_pass':reference_pass,'onnx_target_pass':onnx_pass,'torch_reference_min_mask':min(r['torch_reference']['mask_agreement'] for r in rows) if ref else None,'onnx_min_mask':min(r['onnx_torch']['mask_agreement'] for r in rows),'torch_latency_ms':{'mean':float(np.mean(latency)),'median':float(np.median(latency)),'p95':float(np.percentile(latency,95)),'scope':'batch=1 resize+H2D+AMP networks+detector NMS+D2H; excludes final full-resolution NumPy postprocessing; 3 warmups, CUDA synchronized'},'raw_max_abs_amp':{k:max(r['raw_max_abs_amp'][k] for r in rows) for k in rows[0]['raw_max_abs_amp']},'raw_max_abs_fp32':{k:max(r['raw_max_abs_fp32'][k] for r in rows) for k in rows[0]['raw_max_abs_fp32']},'onnx_diagnosis':'Approval: strict FP32 (b), autocast/TF32 off; AMP FP16 (a) warning only. AGX TensorRT FP16 requires device validation using (a).' ,'rows':rows,'checkpoint_sha256':{k:hashlib.file_digest(open(p.version_dir/f'{k}.pt','rb'),'sha256').hexdigest() for k in ('seg','det')}}
    (out/'verify.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    if not onnx_pass: raise RuntimeError('Strict FP32 ONNX parity (b) failed')
    if ref and not reference_pass: raise RuntimeError('PyTorch reference parity failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--package',default=str(Path(__file__).resolve().parents[1])); parser.add_argument('--version'); parser.add_argument('--images',required=True); parser.add_argument('--provenance'); parser.add_argument('--frames',help='Fixed frame JSON; defaults to selected version manifest'); parser.add_argument('--reference-source'); parser.add_argument('--version-reference',action='store_true'); parser.add_argument('--count',type=int,default=20); parser.add_argument('--output',required=True); parser.add_argument('--smoke',action='store_true'); main(parser.parse_args())
