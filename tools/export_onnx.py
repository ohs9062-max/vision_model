#!/usr/bin/env python3
# 세그멘테이션과 검출 원출력을 고정 배치 FP32 ONNX로 내보낸다.
"""Static batch=1 FP32 network graphs; detector decoding/NMS remains NumPy."""
import argparse,inspect,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from vision_model.infer_torch import TorchPredictor

# 검출기의 디코딩 전 원출력을 내보내기 위해 감싼다.
class DetRaw(torch.nn.Module):
    # DetRaw의 구성 요소와 설정을 초기화한다.
    def __init__(self,model): super().__init__(); self.model=model
    # 검출기의 디코딩 전 원출력을 내보내기 위해 감싼다.
    def forward(self,x): return self.model.head(self.model.neck(self.model.backbone(x)))

# 두 모델을 ONNX로 내보내고 그래프 구조를 검사한다.
def export(package, version=None):
    p=TorchPredictor(package,device='cpu',version=version); report={}
    # 두 모델의 원출력을 내보내고 ONNX 구조를 검사한다.
    for kind in ('seg','det'):
        model=p.models[kind] if kind=='seg' else DetRaw(p.models[kind]).eval()
        x=torch.zeros(1,3,*p.config[kind]['input_size']); path=p.version_dir/f'{kind}.onnx'
        names=['logits'] if kind=='seg' else ['logits','distances','priors']
        kwargs = {'dynamo':False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(model,x,str(path),input_names=['images'],output_names=names,opset_version=17,**kwargs)
        import onnx
        graph=onnx.load(str(path)); onnx.checker.check_model(graph)
        report[kind]={'input':list(x.shape),'outputs':{o.name:[d.dim_value for d in o.type.tensor_type.shape.dim] for o in graph.graph.output},'bytes':path.stat().st_size,'checker':'passed'}
    (p.version_dir/'export.json').write_text(json.dumps(report,indent=2)); return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--package',default=str(Path(__file__).resolve().parents[1])); p.add_argument('--version'); a=p.parse_args(); torch.set_num_threads(4); print(json.dumps(export(a.package,a.version),indent=2))
