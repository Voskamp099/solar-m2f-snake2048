# Matched all-scale control and snake arms, within the existing wall cap.
assert summary['status']=='complete' and len(summary['arms'])==2
assert time.monotonic()-PILOT_STARTED+2400<PILOT_WALL_CAP
summary['status']='all_scale_extension_running'
# Separate feasibility probe; it makes no optimizer updates and does not alter arm inputs.
create_arm('snake',(0,1,2));model.train();model.backbone.eval()
for adapter in [model.sem_seg_head.snake_mask_adapter,*model.sem_seg_head.snake_scale_adapters.values()]:
    adapter.alpha.data.fill_(.01)
CROP_SIZE=2048;rng=random.Random(20261017);torch.cuda.reset_peak_memory_stats()
try:
    full_item,_=sample_training()
    with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):
        full_losses=model([full_item]);full_loss=sum(full_losses.values())
    with torch.enable_grad():full_loss.backward()
    assert torch.isfinite(full_loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    full_probe=dict(crop=2048,base_frozen=True,mode='snake_mask_plus_all_scales',
                    peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,optimizer_updates=0,status='PASS')
    del full_item,full_losses,full_loss
except torch.cuda.OutOfMemoryError:
    full_probe=dict(crop=2048,base_frozen=True,status='OOM',optimizer_updates=0)
    for key in ('full_item','full_losses','full_loss'):
        globals().pop(key,None)
finally:
    summary['preflights'].append(full_probe)
    print('SNAKE_FULL2048_PREFLIGHT',json.dumps(full_probe),flush=True)
    CROP_SIZE=1024;detach_adapters()
for mode in ('conv', 'snake'):
    assert time.monotonic() - PILOT_STARTED + 1200 < PILOT_WALL_CAP, 'Not enough time for another pilot arm'
    config = create_arm(mode, (0, 1, 2))
    model.train()
    model.backbone.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    gate = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith('.alpha')]
    others = [p for n, p in model.named_parameters() if p.requires_grad and (not n.endswith('.alpha'))]
    optimizer = torch.optim.AdamW([dict(params=others, lr=0.0001), dict(params=gate, lr=0.001)], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=64.0)
    rng = random.Random(20261018)
    torch.manual_seed(20261018)
    history = []
    updates = 0
    arm_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, 201):
        assert time.monotonic() - PILOT_STARTED < PILOT_WALL_CAP - 900
        item, filename = sample_training()
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad(), torch.autocast('cuda', dtype=torch.float16):
            losses = model([item])
            loss = sum(losses.values())
        assert torch.isfinite(loss), 'Nonfinite loss'
        previous_scale = scaler.get_scale()
        with torch.enable_grad():
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        updates += int(scaler.get_scale() >= previous_scale)
        history.append(dict(step=step, loss=float(loss.detach()), file_name=filename, gradient_norm=float(norm), alpha=float(gate[0].detach()), scale=scaler.get_scale()))
        del losses, loss, item
        if step == 1 or step % 25 == 0:
            print('SNAKE_TRAIN', mode, step, json.dumps(history[-1]), 'elapsed_s', round(time.monotonic() - arm_started), flush=True)
        if step % 50 == 0:
            checkpoint(OUT / (mode + '_all_scales_last.pth'), config, step, optimizer, scaler, rng, history)
    assert all((torch.equal(v.detach().cpu(), base_weights[k]) for k, v in model.state_dict().items() if '.snake_' not in k)), 'Frozen base changed'
    train_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    model.zero_grad(set_to_none=True)
    metrics = evaluate_arm(mode + '_all_scales')
    checkpoint(OUT / (mode + '_all_scales_last.pth'), config, 200, optimizer, scaler, rng, history, metrics)
    result = dict(mode=mode + '_all_scales', config=config, training_steps=200, optimizer_updates=updates, trainable_parameters=sum((p.numel() for p in params)), training_peak_vram_gib=train_peak, calibration=metrics, elapsed_seconds=round(time.monotonic() - arm_started))
    summary['arms'].append(result)
    (OUT / (mode + '_all_scales_history.json')).write_text(json.dumps(history))
    (OUT / 'pilot_summary.json').write_text(json.dumps(summary, indent=2))
    print('SNAKE_ARM_COMPLETE', json.dumps(result), flush=True)
    del optimizer, scaler, params, gate, others
    detach_adapters()
summary['status']='complete_four_arms'
summary['elapsed_seconds']=round(time.monotonic()-PILOT_STARTED)
(OUT/'pilot_summary.json').write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('/kaggle/working/snake_pilot_20261017.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.iterdir():archive.write(path,path.name)
print('SNAKE_FOUR_ARM_COMPLETE',json.dumps(summary),flush=True)
