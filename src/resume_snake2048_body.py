PARENT=Path('/kaggle/working/snake2048_controls_20261018')
PARENT_SHA={'head_only': '2666428a467377c8bbaaa4069c99e1a0965e65bc88ad0ba18ae5c9f4237de2a5', 'fixed_snake': '0f624da140600ccf94addc983dc5fc2482623df49251ebd601db4d9601d85a1f', 'snake': '0bb41ccd6abfb250a6f03c0313651bbfb3752bdac5b49cab5107ddc504ad2d21'}
parent_schedule=json.loads((PARENT/'training_schedule.json').read_text())
assert len(parent_schedule)==600 and schedule[:600]==parent_schedule
PARENT_SCHEDULE_SHA=hashlib.sha256(json.dumps(parent_schedule).encode()).hexdigest()
for mode,expected in PARENT_SHA.items():
    assert hashlib.sha256((PARENT/(mode+'_last.pth')).read_bytes()).hexdigest()==expected

def restore_arm(mode,path):
    global RESUME_INPUT_SHA
    RESUME_INPUT_SHA=hashlib.sha256(path.read_bytes()).hexdigest()
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['source_checkpoint_sha256']==SOURCE_SHA and cp['arm']==mode and cp['input_size']==2048
    config=create_arm(mode)
    assert config==cp['adapter_config']
    assert hasattr(cp['delta'],'_metadata') and cp['delta']._metadata['sem_seg_head']['version']==2
    expected_missing=set(model.state_dict())-set(cp['delta'])
    report=model.load_state_dict(cp['delta'],strict=False)
    assert not report.unexpected_keys and set(report.missing_keys)==expected_missing
    assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in cp['delta'].items())
    assert [n for n,p in model.named_parameters() if p.requires_grad]==cp['trainable_names']
    for row in cp['history']:
        if not row['optimizer_updated']:continue
        spec={k:row[k] for k in ('file_name','source_image_id','hflip','vflip')}
        assert spec==schedule[row['effective_update']-1]
    optimizer,params=optimizer_for_arm();optimizer.load_state_dict(cp['optimizer'])
    scaler=torch.amp.GradScaler('cuda',init_scale=64.);scaler.load_state_dict(cp['scaler'])
    random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng']);torch.set_rng_state(cp['torch_rng'])
    assert len(cp['cuda_rng'])==torch.cuda.device_count()
    torch.cuda.set_rng_state_all(cp['cuda_rng'])
    return cp,config,optimizer,scaler,params

summary_path=OUT/'summary.json'
if summary_path.exists():
    summary=json.loads(summary_path.read_text())
    assert summary['status'] not in ('complete','stopped_no_calibration_improvement'), 'This continuation already completed; do not rerun it.'
else:
    summary=dict(experiment='native2048_matched_resume250',source_sha256=SOURCE_SHA,
                 origin_step250_sha256=PARENT_SHA,parent_schedule_sha256=PARENT_SCHEDULE_SHA,
                 schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
                 input_size=2048,input_policy='whole native2048, no crop or downsample',
                 targets=[600,1200],rounds=[],preflights=[],status='restoring',
                 baseline_calibration_pq=.4563696594893772,baseline_public_score=.36,
                 parent_calibration_pq=dict(head_only=.45524382357028764,fixed_snake=.4541071485523339,snake=.4555638268539116),
                 trainable_original_module=MASK_PREFIX,training_images=566,calibration_images=70,
                 mask_embed_lr=1e-6,adapter_lr=1e-4,gate_lr=1e-3,weight_decay=.01,
                 wall_cap_seconds=WALL_CAP_SECONDS,review_evaluated=False,submitted=False,
                 continuation_policy='Run all three arms to600. Continue all three to1200 only if at least one600 PQ improves its own250 PQ by>.001. Selection uses calibration only. No CSV submission.',
                 train_seed=TRAIN_SEED,init_seed=INIT_SEED)
write_json(summary_path,summary);emit('RESUME2048_START',summary)
try:
    if len(summary['preflights'])<3:
        for mode in ('head_only','fixed_snake','snake'):
            if any(pr['arm']==mode for pr in summary['preflights']):continue
            path=PARENT/(mode+'_last.pth');cp,config,optimizer,scaler,params=restore_arm(mode,path)
            assert cp['effective_updates']==250
            model.eval();name=sorted(n for n in part if part[n]=='calibration')[0]
            with torch.inference_mode():restored_prediction=candidates(train_dir/name,'native')
            expected=json.loads((PARENT/(mode+'_candidates.json')).read_text())[name]['native']
            assert len(expected)==len(restored_prediction)
            assert all(a['rle']==b['rle'] and a['view']==b['view'] and abs(a['score']-b['score'])<1e-5 for a,b in zip(expected,restored_prediction))
            model.train();model.backbone.eval();item=training_item(schedule[250]);optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):losses=model([item]);loss=sum(losses.values())
            assert torch.isfinite(loss)
            with torch.enable_grad():scaler.scale(loss).backward()
            scaler.unscale_(optimizer);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            probe=dict(arm=mode,status='PASS',restore_step=250,prediction_cache_match=True,
                       next_sample=schedule[250],loss=float(loss.detach()),discarded_optimizer_steps=1,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            summary['preflights'].append(probe);write_json(summary_path,summary);emit('RESUME2048_PREFLIGHT_PASS',probe)
            del item,losses,loss,optimizer,scaler,params,cp;cleanup()
    for target in (600,1200):
        if any(r['target']==target for r in summary['rounds']):continue
        if target==1200:
            round600=next(r for r in summary['rounds'] if r['target']==600)
            improved=any(a['calibration']['PQ']>summary['parent_calibration_pq'][a['arm']]+.001 for a in round600['arms'])
            if not improved:
                summary['status']='stopped_no_calibration_improvement';summary['stop_reason']='No600 arm improved its matched250 PQ by>.001.'
                emit('RESUME2048_EARLY_STOP',dict(reason=summary['stop_reason']));break
            assert time.monotonic()-EXPERIMENT_START+5500<WALL_CAP_SECONDS,'Insufficient time for matched1200 round'
        phase=dict(target=target,arms=[])
        if 'incomplete_round' in summary and summary['incomplete_round']['target']==target:phase=summary['incomplete_round']
        summary['incomplete_round']=phase;summary['status']='training';write_json(summary_path,summary)
        for mode in ('head_only','fixed_snake','snake'):
            if any(a['arm']==mode for a in phase['arms']):continue
            candidate=OUT/(mode+'_last.pth')
            path=candidate if candidate.exists() else PARENT/(mode+'_last.pth')
            cp,config,optimizer,scaler,params=restore_arm(mode,path)
            updates=cp['effective_updates'];attempts=cp['attempts'];history=cp['history'];assert 250<=updates<=target
            emit('RESUME2048_RESTORED',dict(arm=mode,step=updates,target=target,input_checkpoint_sha256=RESUME_INPUT_SHA))
            model.train();model.backbone.eval();arm_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            while updates<target:
                assert time.monotonic()-EXPERIMENT_START<WALL_CAP_SECONDS-1000,'Training wall cap reached'
                assert attempts<target+100,'Too many AMP retries'
                item=training_item(schedule[updates]);optimizer.zero_grad(set_to_none=True);attempts+=1
                with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):losses=model([item]);loss=sum(losses.values())
                assert torch.isfinite(loss),'Nonfinite continuation loss'
                old_scale=scaler.get_scale()
                with torch.enable_grad():scaler.scale(loss).backward()
                scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(optimizer);scaler.update()
                accepted=scaler.get_scale()>=old_scale;updates+=int(accepted)
                row=dict(attempt=attempts,effective_update=updates,optimizer_updated=bool(accepted),loss=float(loss.detach()),gradient_norm=float(norm),scale=scaler.get_scale(),**schedule[updates-int(accepted)])
                if mode!='head_only':row['alpha']=float(model.sem_seg_head.snake_mask_adapter.alpha.detach())
                history.append(row);del item,losses,loss
                if attempts%50==0:emit('RESUME2048_TRAIN',dict(arm=mode,target=target,**row,elapsed_seconds=round(time.monotonic()-arm_started)))
                if accepted and updates%50==0:save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
            integrity=verify_frozen();train_peak=torch.cuda.max_memory_allocated()/1024**3
            save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
            model.zero_grad(set_to_none=True);label=mode+'_step'+str(target);metrics=evaluate_arm(label)
            save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history,metrics)
            shutil.copy2(OUT/(mode+'_last.pth'),OUT/(label+'.pth'))
            write_json(OUT/(label+'_history.json'),history)
            result=dict(arm=mode,config=config,target=target,effective_updates=updates,attempts=attempts,calibration=metrics,
                        integrity=integrity,training_peak_allocated_gib=train_peak,elapsed_seconds=round(time.monotonic()-arm_started))
            phase['arms'].append(result);write_json(summary_path,summary);emit('RESUME2048_ARM_COMPLETE',result)
            del cp,optimizer,scaler,params;cleanup();pack_outputs()
        summary['rounds'].append(phase);summary.pop('incomplete_round',None)
        write_json(summary_path,summary);emit('RESUME2048_ROUND_COMPLETE',phase);pack_outputs()
    if summary['status']!='stopped_no_calibration_improvement':summary['status']='complete'
except Exception as error:
    summary['status']='error';summary['error_type']=type(error).__name__;summary['error']=str(error);traceback.print_exc()
    emit('RESUME2048_ERROR',dict(type=type(error).__name__,message=str(error)))
    if 'history' in globals() and 'optimizer' in globals():save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
finally:
    summary['elapsed_seconds']=round(time.monotonic()-EXPERIMENT_START)
    summary['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(summary_path,summary);emit('RESUME2048_COMPLETE',summary)
    model.eval();model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();pack_outputs()
    print('RESUME2048_ARCHIVE_READY',str(OUT.with_suffix('.zip')),flush=True)
