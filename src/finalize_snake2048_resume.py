from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;O=P/'snake2048_resume_20261018'
s=json.loads((O/'summary.json').read_text());v=json.loads((O/'independent_verification.json').read_text());receipt=json.loads((O/'session_receipt.json').read_text())
assert s['status'] in ('complete','stopped_no_calibration_improvement') and v['status']=='PASS' and receipt['session_off']
assert all(len(r['arms'])==3 for r in s['rounds'])
now=datetime.datetime.now(datetime.timezone.utc).isoformat();all_results=[a for r in s['rounds'] for a in r['arms']];best=max(all_results,key=lambda a:a['calibration']['PQ']);gain=best['calibration']['PQ']-s['baseline_calibration_pq']
status='complete_candidate_not_submitted' if gain>.003 else 'complete_not_promoted'
record=dict(status=status,terminal_status=s['status'],completed_at_utc=now,source_checkpoint_sha256=s['source_sha256'],rounds=s['rounds'],preflights=s['preflights'],elapsed_seconds=s['elapsed_seconds'],stop_reason=s.get('stop_reason','planned rounds complete'),gpu_status='off',verified_gpu_off=True,submitted=False,review_evaluated=False,public_score_note='0.36 is previous baseline submission; no new submission in this continuation.',best_arm=best['arm'],best_step=best['target'],best_local_validation_pq=best['calibration']['PQ'],gain_over_original_baseline=gain,report=str(P/'snake2048_resume_20261018_report.txt'),results=str(O),verification=str(O/'independent_verification.json'),gpu_off_screenshot=str(O/'gpu_off.png'),archive=receipt['downloaded_archive'],quota_display_at_end=receipt.get('quota_display'),quota_hours_consumed=None)
for name in ['cycle_monitor_state.json','resume_state.json']:
 path=P/name;d=json.loads(path.read_text());d['full2048_continuation']={**d.get('full2048_continuation',{}),**record};d.update(status='full2048_continuation_complete_gpu_off',phase='full2048_continuation_complete_gpu_off',gpu_session_status='off',active_experiment=None,active_deadline_utc=None,updated_at_utc=now,last_verified_utc=now,latest_user_instruction='已按用户继续使用剩余免费GPU的授权完成三组匹配续训，保存结果/log并关停。新模型未提交，0.36是旧基线公开分。')
 if name=='cycle_monitor_state.json':d['latest_live_check']=receipt;d['experiment_status']=record['status']
 else:d['next_steps']=['当前GPU已off，没有运行中的任务或自动化。','本次未提交CSV；0.36为此前基线模型已测公开分，不是新蛇形模型成绩。',f'三组已完成匹配续训阶段{[r["target"] for r in s["rounds"]]}；最好本地验证PQ={best["calibration"]["PQ"]:.8f}，相对原基线{gain:+.8f}。','所有阶段结果、完整Notebook和恢复状态已本地保存。不能把本地验证PQ当作Kaggle公开分。']
 path.write_text(json.dumps(d,ensure_ascii=False,indent=2))
f=P/'snake2048_resume_20261018_plan.json';d=json.loads(f.read_text());d.update(record);f.write_text(json.dumps(d,ensure_ascii=False,indent=2))
f=P/'experiment_index_20261017.json';d=json.loads(f.read_text());d.update(status='full2048_continuation_complete_gpu_off',gpu='off',updated_at_utc=now)
for e in d['experiments']:
 if e['id']=='E09_full2048_matched_continuation':e.update(record)
d['logs']['full2048_continuation_report']=record['report'];d['logs']['full2048_continuation_verification']=record['verification'];f.write_text(json.dumps(d,ensure_ascii=False,indent=2))
line=f'\n2026-10-18已完成三组2048续训，阶段{[r["target"] for r in s["rounds"]]}，全部从250步恢复优化器及RNG且采样顺序匹配。最好{best["arm"]} step{best["target"]}，本地验证PQ={best["calibration"]["PQ"]:.8f}，相对原基线{gain:+.8f}。本地独立复算PASS。GPU已off，未提交CSV，0.36仍为此前基线已测公开分。详见snake2048_resume_20261018_report.txt。\n'
for name in ['research_strategy_20261017.txt','alignment_20261017.txt']:
 with (P/name).open('a') as f:f.write(line)
files=[f for f in O.rglob('*') if f.is_file() and f.name!='artifact_sha256.json']+[P/n for n in ['snake2048_resume_run_cell.py','resume_snake2048_body.py','verify_snake2048_resume.py','report_snake2048_resume.py','snake2048_resume_20261018_report.txt','snake2048_resume_executed_source_20261018.ipynb','snake2048_resume_executed_complete_20261018.ipynb']]
manifest={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
(O/'artifact_sha256.json').write_text(json.dumps(dict(created_at_utc=now,sha256=manifest,source_checkpoint_sha256=s['source_sha256'],origin_step250_sha256=s['origin_step250_sha256'],archive_sha256=hashlib.sha256(Path(receipt['downloaded_archive']).read_bytes()).hexdigest()),indent=2))
(P/'CONTINUATION_CURRENT.txt').write_text('2026-10-18续训任务已结束，GPUoff，active_experimentnull。新模型没有提交；0.36来自之前原模型。\n'+line+'\n结果：'+str(O)+'\n报告：'+record['report']+'\n')
print(json.dumps(dict(status=status,gpu='off',best_arm=best['arm'],best_step=best['target'],local_validation_pq=best['calibration']['PQ'],gain=gain,manifest_files=len(manifest)),ensure_ascii=False))
