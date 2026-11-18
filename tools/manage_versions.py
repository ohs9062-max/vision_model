#!/usr/bin/env python3
# 모델 버전을 생성·검증하고 승인된 버전으로 활성 설정을 변경한다.
"""Fail-closed model updates; activation remains a separate operation."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import yaml

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from vision_model.versioning import load_version,validate_postprocess,version_dir
FINAL_PACK=Path('/home/hs/rang/robot_vision3_autolabel/experiments/student_deploy_20261019/final_pack.py')

# 후처리 상수를 한 번 추출하고 지원하는 구조인지 확인한다.
def read_constants(source):
    schema=json.loads((PACKAGE/'tools/postprocess_schema.json').read_text())
    tree=ast.parse(source.read_text())
    declarations={}
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target,ast.Name) and target.id.isupper():
                    if target.id in declarations: raise ValueError(f'Reassigned constant {target.id}: review implementation')
                    declarations[target.id]=node.value
    if set(declarations)!=set(schema['source_constants']): raise ValueError('Unsupported final_pack constant set changed; review package implementation')
    for key,expected in schema['non_postprocess'].items():
        if ast.dump(declarations[key],include_attributes=False)!=expected: raise ValueError(f'Unsupported non-postprocess constant change: {key}')
    values={key:ast.literal_eval(declarations[key]) for key in schema['keys']}
    for key in ('MERGE_CLASSES','FAR_DROP_CLASSES'):
        if not isinstance(values[key],set): raise ValueError(f'Unsupported {key}: expected set')
        values[key]=sorted(values[key])
    for key,value in values.items():
        if type(value).__name__!=schema['types'][key]: raise ValueError(f'Unsupported {key} structure')
    return validate_postprocess(values)

# 지정 알고리즘으로 파일 해시를 계산한다.
def digest(path,kind='md5'):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,kind).hexdigest()

# 새 버전을 복사·내보내기·검증하고 엄격 기준 통과 때 승인한다.
def update(name,seg,det,postprocess_baseline=False,postprocess_config=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name): raise ValueError('Invalid model name')
    # 1) 입력 체크포인트·상수·검증 이미지 확인하기
    source_paths={k:Path(p).resolve(strict=True) for k,p in [('seg',seg),('det',det)]}
    baseline,_,_=load_version(PACKAGE)
    constants=validate_postprocess(yaml.safe_load((baseline/'postprocess.yaml').read_text())) if postprocess_baseline else read_constants(FINAL_PACK)
    if postprocess_config:
        constants=validate_postprocess(yaml.safe_load(Path(postprocess_config).read_text()))
    frames=json.loads((baseline/'verification_frames.json').read_text())
    if len(frames)!=20 or len(set(frames))!=20: raise ValueError('Exactly 20 fixed parity frames required')
    for frame in frames:
        if not Path(frame).is_file(): raise FileNotFoundError(frame)
    version=datetime.date.today().strftime('%Y%m%d')+'_'+name
    target=PACKAGE/'weights'/version
    # 2) 새 버전 폴더와 복사본 만들기
    target.mkdir() # never overwrite an existing bundle
    (target/'INCOMPLETE').write_text('update in progress\n')
    try:
        for kind,path in source_paths.items(): shutil.copy2(path,target/f'{kind}.pt')
        (target/'postprocess.yaml').write_text(yaml.safe_dump(constants,sort_keys=False))
        (target/'verification_frames.json').write_text(json.dumps(frames,indent=2))
        # 3) 모델 설정과 출처 보존하기
        import torch
        configs={}
        for kind in ('seg','det'):
            checkpoint=torch.load(target/f'{kind}.pt',map_location='cpu',weights_only=False)
            configs[kind]=checkpoint['config']
            if checkpoint['config']['num_classes']!={'seg':3,'det':4}[kind]: raise ValueError('Unsupported checkpoint class structure')
            del checkpoint
        (target/'model_config.yaml').write_text(yaml.safe_dump(configs,sort_keys=False))
        post_source=Path(postprocess_config) if postprocess_config else (baseline/'postprocess.yaml' if postprocess_baseline else FINAL_PACK)
        lines=[f'adopted_date: {datetime.date.today().isoformat()}',f'postprocess_source: {post_source}',f'postprocess_source_md5: {digest(post_source)}']
        for kind,path in source_paths.items(): lines.extend([f'{kind}_source: {path}',f'{kind}_md5: {digest(target/f"{kind}.pt")}'])
        (target/'SOURCE.txt').write_text('\n'.join(lines)+'\n')
        # 4) 내보내기 후 엄격 FP32 승인 검사하기
        subprocess.run([sys.executable,str(PACKAGE/'tools/export_onnx.py'),'--package',str(PACKAGE),'--version',version],check=True)
        subprocess.run([sys.executable,str(PACKAGE/'tools/verify_version.py'),'--version',version,'--baseline',baseline.name],check=True)
        report=json.loads((target/'verify.json').read_text())
        if report.get('b',{}).get('passed') is not True: raise RuntimeError('Strict FP32 approval (b) required')
        (target/'INCOMPLETE').unlink()
    except BaseException as error:
        (target/'FAILED').write_text(f'{type(error).__name__}: {error}\n')
        raise
    print(f'검증 통과: {target}\nactive_model 바꾸려면 config/model.yaml 수정 (자동 교체하지 않음)')

# 검증 결과와 파일 해시를 확인하고 활성 버전을 변경한다.
def rollback(version):
    folder,_,_=load_version(PACKAGE,version)
    report=json.loads((folder/'verify.json').read_text())
    if (folder/'FAILED').exists() or (folder/'INCOMPLETE').exists() or report.get('frames')!=20 or report.get('b',{}).get('passed') is not True or report.get('onnx_target_pass') is not True or report.get('torch_reference_pass') is False:
        raise ValueError('Version verification has not passed')
    # Verify artifact identity as well as the saved pass flag.
    # 1) 검증 당시 파일 해시 확인하기
    hashes=report.get('artifact_sha256')
    if not hashes: raise ValueError('Missing verified artifact hashes')
    for name,expected in hashes.items():
        if Path(name).name!=name or digest(folder/name,'sha256')!=expected: raise ValueError(f'Artifact changed: {name}')
    # 2) 활성 버전 한 줄을 원자적으로 변경하기
    path=PACKAGE/'config/model.yaml'; original=path.read_text()
    lines=original.splitlines(keepends=True)
    matches=[i for i,line in enumerate(lines) if re.match(r'^active_model\s*:',line)]
    if len(matches)!=1: raise ValueError('Need exactly one top-level active_model')
    i=matches[0]; lines[i]=f'active_model: {version}\n'
    temporary=path.with_name('model.yaml.tmp')
    with temporary.open('x') as stream:
        stream.write(''.join(lines)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)
    print(f'active_model: {version}')

# 명령행 인자를 해석하고 버전 관리 작업을 실행한다.
def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('update'); p.add_argument('name'); p.add_argument('seg'); p.add_argument('det'); p.add_argument('--postprocess-baseline',action='store_true'); p.add_argument('--postprocess-config')
    p=sub.add_parser('rollback'); p.add_argument('version'); a=parser.parse_args()
    if a.command=='update': update(a.name,a.seg,a.det,a.postprocess_baseline,a.postprocess_config)
    else: rollback(a.version)

if __name__=='__main__': main()
