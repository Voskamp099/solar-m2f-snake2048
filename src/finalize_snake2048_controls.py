from pathlib import Path
import json,hashlib,datetime,shutil
P=Path(__file__).resolve().parent;O=P/'snake2048_controls_20261018'
s=json.loads((O/'summary.json').read_text());v=json.loads((O/'independent_verification.json').read_text());receipt=json.loads((O/'session_receipt.json').read_text())
assert s['status']=='complete_three_arms' and len(s['arms'])==3 and v['status']=='PASS' and receipt['session_off']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
best=max(s['arms'],key=lambda a:a['calibration']['PQ']);gain=best['calibration']['PQ']-s['baseline_calibration_pq']
status='complete_candidate_not_submitted' if gain>.003 else 'complete_not_promoted'
conclusion=dict(status=status,completed_at_utc=now,source_sha256=s['source_sha256'],arms=s['arms'],preflights=s['preflights'],effective_updates_per_arm=s['effective_updates_per_arm'],input_size=2048,training_input='whole native2048, no cropping or resizing',gpu_status='off',verified_gpu_off=True,elapsed_seconds=s['elapsed_seconds'],report=str(P/'snake2048_controls_20261018_report.txt'),results=str(O),independent_verification=str(O/'independent_verification.json'),gpu_off_screenshot=str(O/'gpu_off.png'),archive=receipt['downloaded_archive'],best_arm=best['arm'],best_calibration_pq=best['calibration']['PQ'],gain_over_original_baseline=gain,submitted=False,review_evaluated=False,public_score_unchanged=.36,quota_display_at_end=receipt.get('quota_display'),quota_hours_consumed=None,quota_note='Exact billed hours not inferred from possibly delayed quota display; experiment stayed within bounded free-GPU allowance.')
for name in ['cycle_monitor_state.json','resume_state.json']:
 path=P/name;d=json.loads(path.read_text());d['full2048_controls']={**d.get('full2048_controls',{}),**conclusion};d['active_experiment']=None;d['gpu_session_status']='off';d['active_deadline_utc']=None;d['updated_at_utc']=now;d['last_verified_utc']=now;d['phase']='full2048_controls_complete_gpu_off';d['status']='full2048_controls_complete_gpu_off';d['latest_user_instruction']='已按用户授权完成完整2048显存预检和三组免费GPU对照，结果及log已存本地，GPU会话已关闭。'
 if name=='cycle_monitor_state.json':d['experiment_status']=status;d['latest_live_check']=receipt;d['verified_at_utc']=now
 else:d['next_steps']=['当前没有运行中的任务；三组2048训练及评估、本地独立复算已完成。','保留公开0.36原权重；未生成或提交新CSV。',f'本轮最佳{best["arm"]}，校准PQ={best["calibration"]["PQ"]:.8f}，相对原基线差值{gain:+.8f}；单种子250步小试不代表充分训练上限。','所有检查点包含delta元数据、优化器/scaler及RNG；恢复时从相同基线SHA加载并继续原采样序列。','不恢复旧heartbeat，不Run All；后续研究按用户下一次安排执行。']
 path.write_text(json.dumps(d,ensure_ascii=False,indent=2))
plan=P/'snake2048_controls_20261018_plan.json';d=json.loads(plan.read_text());d.update(conclusion);plan.write_text(json.dumps(d,ensure_ascii=False,indent=2))
idx=P/'experiment_index_20261017.json';d=json.loads(idx.read_text());d['gpu']='off';d['status']='full2048_three_arm_controls_complete';d['updated_at_utc']=now
for e in d['experiments']:
 if e['id']=='E08_full2048_mask_embed_and_snake_controls':e.update(conclusion)
d['logs']['full2048_controls_report']=conclusion['report'];d['logs']['full2048_controls_verification']=conclusion['independent_verification'];idx.write_text(json.dumps(d,ensure_ascii=False,indent=2))
append=f'\n2026-10-18完整2048三组对照完成，输入不裁剪不缩小，统一250次有效更新，仅解冻mask_embed并按组加入共享mask残差分支。head_only={s["arms"][0]["calibration"]["PQ"]:.8f}；fixed_snake={s["arms"][1]["calibration"]["PQ"]:.8f}；snake={s["arms"][2]["calibration"]["PQ"]:.8f}；原基线=.45636966。三组预检峰值allocated约4.40GiB，含优化器状态；本地复算PASS。GPU已off，未提交CSV，公开分仍0.36。详见snake2048_controls_20261018_report.txt。\n'
for name in ['research_strategy_20261017.txt','alignment_20261017.txt']:
 with (P/name).open('a') as f:f.write(append)
files=[f for f in O.rglob('*') if f.is_file() and f.name!='artifact_sha256.json']+[P/f for f in ['snake_feature_adapter.py','train_snake2048_controls.py','snake2048_run_cell.py','verify_snake2048_controls.py','report_snake2048_controls.py','snake2048_controls_20261018_report.txt','snake2048_executed_source_20261018.ipynb','snake2048_executed_complete_20261018.ipynb']]
manifest={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
(O/'artifact_sha256.json').write_text(json.dumps(dict(created_at_utc=now,sha256=manifest,source_checkpoint_sha256=s['source_sha256'],downloaded_archive_sha256=hashlib.sha256(Path(receipt['downloaded_archive']).read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(status=status,gpu='off',best_arm=best['arm'],best_calibration_PQ=best['calibration']['PQ'],gain=gain,manifest_files=len(manifest)),ensure_ascii=False))
