import datetime, traceback

SOURCE = Path('/kaggle/working/finetune_results/model_best_finetune.pth')
SOURCE_SHA = '6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
DELTA = Path('/kaggle/working/snake2048_resume_20261018/snake_step1200.pth')
DELTA_SHA = 'ca87077f721356018594b3f0fe4cf3bea562991c5f3e484347782cfa7946ad75'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
assert hashlib.sha256(DELTA.read_bytes()).hexdigest() == DELTA_SHA
base = torch.load(SOURCE, map_location='cpu', weights_only=True)
model.load_state_dict(base['model'], strict=True)
cp = torch.load(DELTA, map_location='cpu', weights_only=False)
assert cp['arm'] == 'snake' and cp['effective_updates'] == 1200 and cp['input_size'] == 2048
assert cp['source_checkpoint_sha256'] == SOURCE_SHA
saved_config = cp['adapter_config']
config = attach_feature_adapters(model, **{k:saved_config[k] for k in ('mode','mask_features','scales','bottleneck','kernel_size')})
assert config == saved_config
assert cp['delta']._metadata['sem_seg_head']['version'] == 2
expected_missing = set(model.state_dict()) - set(cp['delta'])
loaded = model.load_state_dict(cp['delta'], strict=False)
assert not loaded.unexpected_keys and set(loaded.missing_keys) == expected_missing
assert all(torch.equal(model.state_dict()[k].detach().cpu(), v) for k,v in cp['delta'].items())
assert all(torch.equal(v.detach().cpu(), base['model'][k]) for k,v in model.state_dict().items() if k not in cp['delta'])
del base, cp
gc.collect()
model.requires_grad_(False).eval()
assert all(not m.training for m in model.modules())

def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def emit(marker, data):
    print(marker, json.dumps(data), flush=True)
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(marker=marker, data=data, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')

def pack_outputs():
    archive = OUT.with_suffix('.zip')
    temp = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.suffix != '.tmp': z.write(p,p.name)
    temp.replace(archive)

summary = dict(experiment='best_snake1200_test_inference', status='preflight',
               base_checkpoint_sha256=SOURCE_SHA, delta_checkpoint_sha256=DELTA_SHA,
               adapter_config=config, selected_step=1200, local_calibration_pq=.46036319475580045,
               previous_best_public_score=.36, new_public_score=None, submitted=False,
               input_size=2048, test_policy='whole native2048; FP32; native/hflip/vflip/rot180',
               postprocessing='>=2 views IoU>.5; mean score>=.9; area>=256; greedy nonoverlap before filtering',
               training_performed=False, free_gpu_only=True, wall_cap_seconds=LIMIT_SECONDS,
               device=torch.cuda.get_device_name(0), started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write_json(OUT/'provenance.json', summary)
try:
    reference = json.loads(Path('/kaggle/working/snake2048_resume_20261018/snake_step1200_candidates.json').read_text())
    probe_name = sorted(reference)[0]
    actual = candidates(train_dir/probe_name, 'native')
    expected = reference[probe_name]['native']
    assert len(actual) == len(expected)
    assert all(a['rle']==b['rle'] and a['view']==b['view'] and abs(a['score']-b['score'])<1e-6 for a,b in zip(actual,expected))
    summary['preflight'] = dict(status='PASS', restored_prediction_cache_match=True, image=probe_name, candidates=len(actual))
    emit('SNAKE_TEST_PREFLIGHT_PASS',summary['preflight'])
    del reference, actual, expected
    names = sorted(p.name for p in test_dir.glob('*.jpeg'))
    assert len(names) == 180
    cache_path = OUT/'test_candidates.json'
    if cache_path.exists():
        prior=json.loads((OUT/'cache_identity.json').read_text())
        assert prior == dict(base_sha256=SOURCE_SHA,delta_sha256=DELTA_SHA,views=list(VIEWS))
        cache=json.loads(cache_path.read_text())
    else: cache={}
    write_json(OUT/'cache_identity.json',dict(base_sha256=SOURCE_SHA,delta_sha256=DELTA_SHA,views=list(VIEWS)))
    assert set(cache) <= set(names)
    summary['status']='inference'
    write_json(OUT/'summary.json',summary)
    for i,name in enumerate(names,1):
        cache.setdefault(name,{})
        for view in VIEWS:
            if view not in cache[name]:
                cache[name][view]=candidates(test_dir/name,view)
                write_json(cache_path,cache)
        if i%10==0 or i==len(names):
            summary.update(processed_test_images=i,elapsed_seconds=round(time.monotonic()-RUN_STARTED))
            write_json(OUT/'summary.json',summary)
            emit('SNAKE_TEST_PROGRESS',dict(images=i,total=180,elapsed_seconds=summary['elapsed_seconds']))
            pack_outputs()
    assert set(cache)==set(names) and all(set(v)==set(VIEWS) for v in cache.values())
    rows=[];image_counts={};predictions={}
    for i,name in enumerate(names,1):
        selected=[p for p in disjoint(consensus(cache[name])) if p['score']>=.9 and p['area']>=256]
        predictions[name]=selected;image_counts[name]=len(selected)
        rows.extend((Path(name).stem+'_'+str(k),p['rle']['counts']) for k,p in enumerate(selected,1))
        if i%30==0:emit('SNAKE_TEST_POSTPROCESS',dict(images=i,total=180))
    assert rows and len({r[0] for r in rows})==len(rows)
    with (OUT/'submission_snake1200.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows(rows)
    write_json(OUT/'test_predictions.json',predictions)
    write_json(OUT/'test_names.json',names)
    summary.update(status='complete', processed_test_images=180,predicted_instances=len(rows),
                   empty_test_images=sum(n==0 for n in image_counts.values()),test_image_counts=image_counts,
                   submission_sha256=hashlib.sha256((OUT/'submission_snake1200.csv').read_bytes()).hexdigest(),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
except Exception as error:
    summary.update(status='error',error_type=type(error).__name__,error=str(error))
    traceback.print_exc()
    emit('SNAKE_TEST_ERROR',dict(type=type(error).__name__,message=str(error)))
finally:
    summary.update(elapsed_seconds=round(time.monotonic()-RUN_STARTED),completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(OUT/'summary.json',summary)
    emit('SNAKE_TEST_COMPLETE',summary)
    gc.collect();torch.cuda.empty_cache();pack_outputs()
    print('SNAKE_TEST_ARCHIVE_READY',str(OUT.with_suffix('.zip')),flush=True)
