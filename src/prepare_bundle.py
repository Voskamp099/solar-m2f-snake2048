from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/raelynn/Downloads/solar_project')
BUNDLE = ROOT / 'bundle'
BUNDLE.mkdir(exist_ok=True)
for name in ('solar_runtime.py', 'solar_inference.py', 'solar_evaluation.py'):
    shutil.copy2(SOURCE / name, BUNDLE / name)
RUN = SOURCE / 'outputs_1002/r50_2048_b1_20261002'
shutil.copy2(RUN / 'config.yaml', BUNDLE / 'config.yaml')
evaluation = json.loads((RUN / 'inference/official_pq.json').read_text())
validation = [{k: row[k] for k in ('source_image_id', 'file_name', 'n_gt')}
              for row in evaluation['rows']]
(BUNDLE / 'validation_records.json').write_text(json.dumps(validation, indent=2))
weight = RUN / 'model_best.pth'
with weight.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
assert digest == '7b540a61c3903db390c676d4a37d808a9320d4ef7b1186e2696ae882fdf9e2ed'
provenance = dict(checkpoint_sha256=digest, iteration=36999,
                  historical_validation_pq=0.309437444407131,
                  training_authorized=False, score_threshold=0.5, image_size=2048)
lean = BUNDLE / 'model_best.pth'
if lean.exists():
    weight = lean
    with lean.open('rb') as f:
        provenance['checkpoint_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    provenance['original_checkpoint_sha256'] = digest
    provenance['model_tensors_identical_to_original'] = True
for repo in ('Mask2Former', 'detectron2'):
    provenance[repo + '_commit'] = subprocess.check_output(
        ['git', '-C', str(ROOT / 'vendor' / repo), 'rev-parse', 'HEAD'], text=True).strip()
(BUNDLE / 'provenance.json').write_text(json.dumps(provenance, indent=2))
with zipfile.ZipFile(ROOT / 'solar_m2f_best_inference_20261016.zip', 'w',
                     compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    z.write(weight, 'model_best.pth')
    for p in sorted(BUNDLE.iterdir()):
        if p.is_file() and p.name != 'model_best.pth':
            z.write(p, p.name)
    for repo in ('Mask2Former', 'detectron2'):
        for p in sorted((ROOT / 'vendor' / repo).rglob('*')):
            rel = p.relative_to(ROOT / 'vendor')
            if p.is_file() and '.git' not in rel.parts and '__pycache__' not in rel.parts:
                z.write(p, str(Path('vendor') / rel))
print(json.dumps(provenance, indent=2))
print('bundle_bytes', (ROOT / 'solar_m2f_best_inference_20261016.zip').stat().st_size)
