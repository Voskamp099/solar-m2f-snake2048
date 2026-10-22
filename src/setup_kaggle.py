import os, sys, time, json, shutil, subprocess
from pathlib import Path

RUN_STARTED = time.monotonic()
INPUTS = Path('/kaggle/input')
candidates = list(INPUTS.rglob('provenance.json'))
candidates = [p for p in candidates if (p.parent / 'model_best.pth').is_file()]
assert len(candidates) == 1, candidates
ASSETS = candidates[0].parent
RUNTIME = Path('/kaggle/working/solar_m2f_runtime')
RUNTIME.mkdir(exist_ok=True)
for name in ['Mask2Former', 'detectron2']:
    target = RUNTIME / name
    if not target.exists():
        shutil.copytree(ASSETS / 'vendor' / name, target)
print('ASSETS:', ASSETS, flush=True)
print('Python:', sys.version, flush=True)
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
assert torch.cuda.is_available(), 'A free Kaggle GPU is required; no CPU full run.'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'ninja', 'pycocotools', 'timm==0.9.16', 'fvcore==0.1.5.post20221221',
    'iopath==0.1.9', 'yacs', 'omegaconf>=2.1,<2.4', 'hydra-core',
    'termcolor', 'tabulate', 'cloudpickle', 'tensorboard', 'shapely',
    'scipy', 'submitit', 'black', 'packaging'], check=True, timeout=900)
# The Mask2Former R50 path needs no Detectron2 CUDA-specific operators.
# Build its normal CPU extension; deformable attention uses the existing
# project's differentiable PyTorch reference implementation during inference.
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MAX_JOBS='2')
log_path = Path('/kaggle/working/detectron2_install.log')
print('Building Detectron2 CPU extension; log:', log_path, flush=True)
with log_path.open('w') as log:
    result = subprocess.run([sys.executable, '-m', 'pip', 'install',
        '--no-build-isolation', '--no-deps', str(RUNTIME / 'detectron2')],
        env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
print(log_path.read_text()[-8000:], flush=True)
assert result.returncode == 0, 'See detectron2_install.log'
sys.path[:0] = [str(ASSETS), str(RUNTIME / 'Mask2Former')]
from solar_runtime import install_msda_backend
BACKEND = install_msda_backend('torch')
import detectron2, mask2former
print('Runtime ready:', detectron2.__version__, BACKEND, flush=True)
print('SETUP_SECONDS:', round(time.monotonic() - RUN_STARTED), flush=True)
