"""Verify portable source/model integrity before starting the local server."""
import argparse,ast,hashlib,json,os,sys
from pathlib import Path

def verify(root):
 root=Path(root).resolve()
 manifest=json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
 failures=[]
 for name,expected in manifest['sha256'].items():
  path=(root/name).resolve()
  if root not in path.parents:failures.append(f'Unsafe manifest path: {name}');continue
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
   failures.append(f'Integrity mismatch: {name}')
  elif path.suffix=='.py':
   try:ast.parse(path.read_text(encoding='utf-8-sig'),filename=name)
   except SyntaxError as error:failures.append(f'Unreadable Python: {error}')
 if failures:raise RuntimeError('\n'.join(failures))
 return manifest

def environment(root):
 site=Path(r'C:\OralWisdom_venv\Lib\site-packages')
 if site.exists():sys.path.insert(0,str(site))
 import importlib.metadata as metadata
 import torch,cv2,flask
 sys.path.insert(0,str(root/'restoration_runtime'))
 from pipeline.brushnet import BrushNetModel,StableDiffusionBrushNetPipeline
 from huggingface_hub import hf_hub_download
 cache={}
 for model in (os.environ.get('ORALWISDOM_BASE_MODEL','runwayml/stable-diffusion-v1-5'),
               os.environ.get('ORALWISDOM_BRUSHNET_MODEL','Sanster/brushnet_segmentation_mask')):
  configs=['config.json'] if model==os.environ.get('ORALWISDOM_BRUSHNET_MODEL','Sanster/brushnet_segmentation_mask') else [
   'model_index.json','scheduler/scheduler_config.json','tokenizer/merges.txt','tokenizer/vocab.json',
   'tokenizer/tokenizer_config.json','text_encoder/config.json','unet/config.json','vae/config.json']
  files=[hf_hub_download(model,name,local_files_only=True) for name in configs]
  weights=['diffusion_pytorch_model.fp16.safetensors'] if len(configs)==1 else [
   'text_encoder/model.fp16.safetensors','unet/diffusion_pytorch_model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors']
  files.extend(hf_hub_download(model,name,local_files_only=True) for name in weights)
  cache[model]=files
 if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in this Python environment')
 return {'python':sys.version,'gpu':torch.cuda.get_device_name(0),'cached_models':cache,
  'packages':{name:metadata.version(name) for name in ('torch','diffusers','transformers','accelerate','ultralytics','Flask')}}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
 parser.add_argument('--environment',action='store_true')
 args=parser.parse_args();manifest=verify(args.root)
 print(f"Integrity passed: {len(manifest['sha256'])} files")
 if args.environment:print(json.dumps(environment(args.root),indent=2))
if __name__=='__main__':main()
