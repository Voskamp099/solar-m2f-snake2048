import csv, gc, hashlib, json, time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mu
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.data import detection_utils as utils, transforms as T
from mask2former import add_maskformer2_config
from solar_inference import install_chunked_instance_inference
from solar_evaluation import prediction_rles, mask_rle, pq_counts, summarize_counts

OUT = Path('/kaggle/working')
provenance = json.loads((ASSETS / 'provenance.json').read_text())
weight_path = ASSETS / 'model_best.pth'
with weight_path.open('rb') as f:
    actual_hash = hashlib.file_digest(f, 'sha256').hexdigest()
assert actual_hash == provenance['checkpoint_sha256']
torch.set_grad_enabled(False)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.set_new_allowed(True)
cfg.merge_from_file(str(ASSETS / 'config.yaml'))
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.WEIGHTS = str(weight_path)
cfg.DATASETS.TRAIN = ('solar_inference_runtime',)
cfg.DATASETS.TEST = ()
assert cfg.INPUT.MIN_SIZE_TEST == cfg.INPUT.MAX_SIZE_TEST == 2048
model = build_model(cfg)
checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
assert checkpoint['iteration'] == 36999
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
gc.collect()
model.requires_grad_(False).eval()
install_chunked_instance_inference(model, 0.5)
aug = T.ResizeShortestEdge([2048, 2048], 2048)
USE_AMP = False
test_dirs = list(INPUTS.rglob('test_images'))
assert len(test_dirs) == 1, test_dirs
TEST_DIR = test_dirs[0]
DATA_ROOT = TEST_DIR.parent.parent
TRAIN_DIR = DATA_ROOT / 'train/train_images'
TRAIN_JSON = DATA_ROOT / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
raw = json.loads(TRAIN_JSON.read_text())
assert hashlib.sha256(TRAIN_JSON.read_bytes()).hexdigest() == '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
source_images = {im['id']: im for im in raw['images']}
gt = defaultdict(list)
for a in raw['annotations']:
    gt[a['image_id']].append(a)

def predict(path):
    assert not model.training and not torch.is_grad_enabled()
    assert time.monotonic() - RUN_STARTED < 3.5 * 3600, 'Stop before 4-hour budget.'
    image = utils.read_image(str(path), format=cfg.INPUT.FORMAT)
    h, w = image.shape[:2]
    assert (h, w) == (2048, 2048)
    transformed = aug.get_transform(image).apply_image(image)
    x = torch.as_tensor(transformed.astype('float32').transpose(2, 0, 1))
    ctx = torch.autocast('cuda', dtype=torch.float16) if USE_AMP else nullcontext()
    with torch.inference_mode(), ctx:
        result = model([{'image': x, 'height': h, 'width': w}])[0]
    rles = prediction_rles(result['instances'], 0.5)
    for rle in rles:
        assert rle['size'] == [2048, 2048]
        assert int(mu.area(rle)) > 0
    del result, x
    return rles

tests = sorted(TEST_DIR.glob('*.jpeg'))
assert len(tests) == 180
print('SMOKE: original 2048 input, first image', tests[0].name, flush=True)
try:
    smoke = predict(tests[0])
except torch.OutOfMemoryError:
    gc.collect()
    torch.cuda.empty_cache()
    USE_AMP = True
    print('FP32 smoke exceeded VRAM; trying AMP at the SAME 2048 resolution.', flush=True)
    smoke = predict(tests[0])
print('SMOKE_OK', len(smoke), 'instances;', 'AMP', USE_AMP,
      'peak_GiB', round(torch.cuda.max_memory_allocated()/1024**3, 3), flush=True)

records = json.loads((ASSETS / 'validation_records.json').read_text())
assert len(records) == 229 and len({r['file_name'] for r in records}) == 141
cache_path = OUT / 'validation_predictions.json'
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
names = sorted({r['file_name'] for r in records})
t0 = time.monotonic()
for i, name in enumerate(names, 1):
    if name not in cache:
        cache[name] = predict(TRAIN_DIR / name)
        cache_path.write_text(json.dumps(cache))
    if i % 10 == 0 or i == len(names):
        print('VALIDATION', i, '/', len(names), 'elapsed_s', round(time.monotonic()-t0), flush=True)
rows = []
for record in records:
    sid = record['source_image_id']
    im = source_images[sid]
    assert im['file_name'] == record['file_name']
    assert len(gt[sid]) == record['n_gt']
    truth = [mask_rle(a['segmentation'], 2048, 2048) for a in gt[sid]]
    pred = cache[im['file_name']]
    iou = mu.iou(truth, pred, [0]*len(pred)) if truth and pred else np.zeros((len(truth), len(pred)))
    rows.append(dict(source_image_id=sid, file_name=im['file_name'], **pq_counts(iou)))
metrics = summarize_counts(rows)
(OUT / 'validation_metrics.json').write_text(json.dumps(dict(metrics=metrics, rows=rows), indent=2))
print('VALIDATION_PQ', json.dumps(metrics), flush=True)
assert metrics['PQ'] >= .27, 'Validation regressed materially; stop before generating submission.'

cache_path = OUT / 'test_predictions.json'
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
cache.setdefault(tests[0].name, smoke)
t0 = time.monotonic()
for i, path in enumerate(tests, 1):
    if path.name not in cache:
        cache[path.name] = predict(path)
    cache_path.write_text(json.dumps(cache))
    if i % 10 == 0 or i == len(tests):
        print('TEST', i, '/', len(tests), 'elapsed_s', round(time.monotonic()-t0), flush=True)
assert set(cache) == {p.name for p in tests}
rows = [(f'{Path(name).stem}_{k}', rle['counts'])
        for name in sorted(cache) for k, rle in enumerate(cache[name], 1)]
assert rows and len({r[0] for r in rows}) == len(rows)
with (OUT / 'submission_m2f_best_2048.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filament_id', 'segmentation_rle'])
    writer.writerows(rows)
provenance.update(backend=BACKEND, amp=USE_AMP, validation=metrics,
    processed_test_images=len(cache), empty_test_images=sum(not v for v in cache.values()),
    predicted_instances=len(rows), training_performed=False,
    elapsed_seconds=round(time.monotonic()-RUN_STARTED),
    peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
    gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
(OUT / 'run_result.json').write_text(json.dumps(provenance, indent=2))
print('INFERENCE_COMPLETE', json.dumps(provenance, indent=2), flush=True)
