
import ast,copy,random,warnings,traceback
from collections import OrderedDict
from detectron2.structures import Instances
warnings.filterwarnings('ignore',category=FutureWarning)
PILOT_STARTED=time.monotonic()
PILOT_WALL_CAP=5400
SOURCE=Path('/kaggle/working/finetune_results/model_best_finetune.pth')
SOURCE_SHA='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
base=torch.load(SOURCE,map_location='cpu',weights_only=True)
model.load_state_dict(base['model'],strict=True)
base_weights=base['model']
del base
training_root=Path('/kaggle/working/training_images')
if not (training_root/'training_manifest.json').exists():
    archives=list(Path('/kaggle/input').rglob('training_images.bin'));assert len(archives)==1
    training_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as z:z.extractall(training_root)
manifest=json.loads((training_root/'training_manifest.json').read_text())
train_names=manifest['train_images'];assert len(train_names)==566 and not set(train_names)&set(part)
train_records=defaultdict(list)
for image in raw['images']:
    if image['file_name'] in set(train_names):train_records[image['file_name']].append(image['id'])
CROP_SIZE=1024
def sample_training():
    name = rng.choice(train_names)
    sid = rng.choice(train_records[name])
    image = utils.read_image(str(training_root / 'train_images' / name), format=cfg.INPUT.FORMAT)
    y, x = (rng.randrange(2048 - CROP_SIZE + 1), rng.randrange(2048 - CROP_SIZE + 1))
    image = image[y:y + CROP_SIZE, x:x + CROP_SIZE]
    masks = []
    for a in gt[sid]:
        mask = mu.decode(mask_rle(a['segmentation'], 2048, 2048))[y:y + CROP_SIZE, x:x + CROP_SIZE]
        if mask.any():
            masks.append(mask)
    masks = np.stack(masks) if masks else np.zeros((0, CROP_SIZE, CROP_SIZE), np.uint8)
    if rng.random() < 0.5:
        image = image[:, ::-1]
        masks = masks[:, :, ::-1]
    if rng.random() < 0.5:
        image = image[::-1]
        masks = masks[:, ::-1, :]
    target = Instances((CROP_SIZE, CROP_SIZE))
    target.gt_classes = torch.zeros(len(masks), dtype=torch.int64)
    target.gt_masks = torch.from_numpy(np.ascontiguousarray(masks)).bool()
    return ({'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))), 'instances': target, 'height': CROP_SIZE, 'width': CROP_SIZE}, name)

def detach_adapters():
    h=model.sem_seg_head
    if hasattr(h,'_snake_feature_handle'):
        h._snake_feature_handle.remove()
        del h._snake_feature_handle
        del h.snake_mask_adapter,h.snake_scale_adapters
        del h._snake_adapter_config
    model.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()

def create_arm(mode,scales=()):
    detach_adapters()
    model.load_state_dict(base_weights,strict=True)
    model.requires_grad_(False)
    torch.manual_seed(20261017)
    config=attach_feature_adapters(model,mode=mode,mask_features=True,scales=scales,bottleneck=32,kernel_size=9)
    return config

def checkpoint(path,config,step,optimizer,scaler,rng,history,metrics=None):
    adapter_state=OrderedDict((k,v.detach().cpu()) for k,v in model.state_dict().items() if '.snake_' in k)
    adapter_state._metadata=copy.deepcopy(model.state_dict()._metadata)
    data=dict(adapters=adapter_state,adapter_config=config,step=step,optimizer=optimizer.state_dict(),
              scaler=scaler.state_dict(),python_rng=rng.getstate(),numpy_rng=np.random.get_state(),
              torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
              source_checkpoint_sha256=SOURCE_SHA,source_checkpoint=str(SOURCE),history=history,metrics=metrics,
              format='adapter_delta_v1; load original base first, attach adapters, then load adapter delta',
              quota_wall_cap_seconds=PILOT_WALL_CAP)
    tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)

def evaluate_arm(label):
    model.eval();cache={}
    selected=sorted(n for n in part if part[n]=='calibration')
    with torch.inference_mode():
        for idx,name in enumerate(selected,1):
            assert time.monotonic()-PILOT_STARTED<PILOT_WALL_CAP
            cache[name]={view:candidates(train_dir/name,view) for view in VIEWS}
            if idx%10==0:print('SNAKE_EVAL_PROGRESS',label,idx,'/70',flush=True)
    (OUT/(label+'_candidates.json')).write_text(json.dumps(cache))
    preds={name:[p['rle'] for p in disjoint(consensus(views)) if p['score']>=.9 and p['area']>=256]
           for name,views in cache.items()}
    rows=[]
    for record in records:
        name=record['file_name']
        if name not in preds:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[record['source_image_id']]]
        pred=preds[name]
        mat=mu.iou(truth,pred,[0]*len(pred)) if pred else np.zeros((len(truth),0))
        rows.append(pq_counts(mat))
    return summarize_counts(rows)

summary=dict(source_sha256=SOURCE_SHA,baseline_calibration_pq=.4563696594893772,
             crop=1024,steps_per_arm=200,evaluation='70 calibration physical images only; locked2048FP32/4view/.90/256',
             preflights=[],arms=[],training='base frozen; newly added residual adapters only',
             submitted=False,review_evaluated=False)
print('SNAKE_PILOT_START',json.dumps(summary),flush=True)
try:
    # Verify attachment preserves the actual pretrained full2048 inference.
    name=sorted(n for n in part if part[n]=='calibration')[0]
    model.eval()
    with torch.inference_mode():baseline_candidates=candidates(train_dir/name,'native')
    create_arm('snake',(0,1,2));model.eval()
    with torch.inference_mode():identity_candidates=candidates(train_dir/name,'native')
    assert baseline_candidates==identity_candidates,'Zero-gate identity regression'
    summary['actual_2048_zero_gate_identity']=True
    # Probe all spatial sites with an active gate and true backwards, no updates.
    model.train();model.backbone.eval();rng=random.Random(20261017)
    for adapter in [model.sem_seg_head.snake_mask_adapter,*model.sem_seg_head.snake_scale_adapters.values()]:
        adapter.alpha.data.fill_(.01)
    sample,_=sample_training();torch.cuda.reset_peak_memory_stats()
    with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):
        losses=model([sample]);loss=sum(losses.values())
    with torch.enable_grad():loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    probe=dict(mode='snake_mask_plus_all_three_scales',loss=float(loss.detach()),
               peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,optimizer_updates=0)
    summary['preflights'].append(probe);print('SNAKE_ALL_SCALE_PREFLIGHT_PASS',json.dumps(probe),flush=True)
    del losses,loss,sample
    detach_adapters()
    for mode in ('conv','snake'):
        assert time.monotonic()-PILOT_STARTED+1200<PILOT_WALL_CAP,'Not enough time for another pilot arm'
        config=create_arm(mode);model.train();model.backbone.eval()
        params=[p for p in model.parameters() if p.requires_grad]
        gate=[p for n,p in model.named_parameters() if p.requires_grad and n.endswith('.alpha')]
        others=[p for n,p in model.named_parameters() if p.requires_grad and not n.endswith('.alpha')]
        optimizer=torch.optim.AdamW([dict(params=others,lr=1e-4),dict(params=gate,lr=1e-3)],weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=64.)
        rng=random.Random(20261018);torch.manual_seed(20261018)
        history=[];updates=0;arm_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for step in range(1,201):
            assert time.monotonic()-PILOT_STARTED<PILOT_WALL_CAP-900
            item,filename=sample_training();optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):
                losses=model([item]);loss=sum(losses.values())
            assert torch.isfinite(loss),'Nonfinite loss'
            previous_scale=scaler.get_scale()
            with torch.enable_grad():scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            scaler.step(optimizer);scaler.update()
            updates+=int(scaler.get_scale()>=previous_scale)
            history.append(dict(step=step,loss=float(loss.detach()),file_name=filename,
                                gradient_norm=float(norm),alpha=float(gate[0].detach()),scale=scaler.get_scale()))
            del losses,loss,item
            if step==1 or step%25==0:
                print('SNAKE_TRAIN',mode,step,json.dumps(history[-1]),'elapsed_s',round(time.monotonic()-arm_started),flush=True)
            if step%50==0:checkpoint(OUT/(mode+'_last.pth'),config,step,optimizer,scaler,rng,history)
        assert all(torch.equal(v.detach().cpu(),base_weights[k]) for k,v in model.state_dict().items() if '.snake_' not in k),'Frozen base changed'
        train_peak=torch.cuda.max_memory_allocated()/1024**3
        model.zero_grad(set_to_none=True)
        metrics=evaluate_arm(mode)
        checkpoint(OUT/(mode+'_last.pth'),config,200,optimizer,scaler,rng,history,metrics)
        result=dict(mode=mode,config=config,training_steps=200,optimizer_updates=updates,
                    trainable_parameters=sum(p.numel() for p in params),training_peak_vram_gib=train_peak,
                    calibration=metrics,elapsed_seconds=round(time.monotonic()-arm_started))
        summary['arms'].append(result)
        (OUT/(mode+'_history.json')).write_text(json.dumps(history))
        (OUT/'pilot_summary.json').write_text(json.dumps(summary,indent=2))
        print('SNAKE_ARM_COMPLETE',json.dumps(result),flush=True)
        del optimizer,scaler,params,gate,others
        detach_adapters()
    summary['status']='complete'
except Exception as error:
    summary['status']='error';summary['error_type']=type(error).__name__;summary['error']=str(error)
    traceback.print_exc()
finally:
    summary['elapsed_seconds']=round(time.monotonic()-PILOT_STARTED)
    (OUT/'pilot_summary.json').write_text(json.dumps(summary,indent=2))
    with zipfile.ZipFile('/kaggle/working/snake_pilot_20261017.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.iterdir():archive.write(path,path.name)
    model.eval();model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    print('SNAKE_PILOT_COMPLETE',json.dumps(summary),flush=True)
