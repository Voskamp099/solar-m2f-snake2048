from pathlib import Path
import datetime, hashlib, json
P=Path(__file__).resolve().parent;O=P/'snake_best_submission_20261018'
s=json.loads((O/'summary.json').read_text());v=json.loads((O/'csv_verification.json').read_text())
gpu=json.loads((O/'session_receipt.json').read_text());r=json.loads((O/'submission_receipt.json').read_text())
assert s['status']=='complete' and v['status']=='PASS' and gpu['session_off']
assert r['status']=='Complete' and r['file_name']=='submission_snake1200.csv'
assert r['submission_sha256']==v['submission_sha256']==s['submission_sha256']
score=float(r['public_score']);now=datetime.datetime.now(datetime.timezone.utc).isoformat()
record=dict(status='submitted_scored_gpu_off',completed_at_utc=now,submitted=True,public_score=score,
            previous_best_public_score=.36,public_score_change=score-.36,free_gpu_only=True,gpu_status='off',
            checkpoint=str(P/'snake2048_resume_20261018/snake_step1200.pth'),delta_sha256=s['delta_checkpoint_sha256'],
            base_checkpoint=str(P/'finetune_results/model_best_finetune.pth'),base_sha256=s['base_checkpoint_sha256'],
            input_size=2048,selected_step=1200,local_calibration_pq=s['local_calibration_pq'],
            review_evaluated=False,training_performed=False,processed_test_images=180,
            predicted_instances=s['predicted_instances'],submission=str(O/r['file_name']),
            submission_sha256=r['submission_sha256'],report=str(O/'result_report.txt'),
            score_screenshot=str(O/'kaggle_public_score.png'),gpu_off_screenshot=str(O/'gpu_off.png'),
            elapsed_inference_seconds=s['elapsed_seconds'],score_precision='As displayed by Kaggle; no hidden precision inferred')
plan=json.loads((O/'local_plan.json').read_text());plan.update(record);(O/'local_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
for name in ['cycle_monitor_state.json','resume_state.json']:
    f=P/name;d=json.loads(f.read_text());d.update(status=record['status'],phase=record['status'],
        snake_best_submission=plan,gpu_session_status='off',active_experiment=None,active_deadline_utc=None,
        latest_submission=record,best_verified_public_score=max(.36,score),updated_at_utc=now,last_verified_utc=now)
    d['next_steps']=[f'蛇形step1200已提交并完成评分，公开分{score:g}；旧版最佳0.36。','GPU已关闭，结果、CSV、日志及源码已保存本地。','本地校准PQ与公开分不能直接换算；没有评估71图复用验证子集，没有根据公开分反复调参。']
    f.write_text(json.dumps(d,ensure_ascii=False,indent=2))
f=P/'experiment_index_20261017.json';d=json.loads(f.read_text());d.update(status=record['status'],gpu='off',updated_at_utc=now)
entry=dict(id='E10_best_snake1200_public_submission',**record)
d['experiments']=[e for e in d['experiments'] if e['id']!=entry['id']]+[entry]
if score>d['current_best']['public_score']:
    d.setdefault('previous_best_models',[]).append(d['current_best'])
    d['current_best']=dict(architecture='Mask2Former / ResNet-50 + dynamic snake shared-mask residual adapter',
        checkpoint=record['base_checkpoint'],checkpoint_sha256=record['base_sha256'],delta_checkpoint=record['checkpoint'],
        delta_sha256=record['delta_sha256'],adapter_config=s['adapter_config'],finetune_best_step=1200,
        input_training='full native2048; only mask_embed and shared adapter trained',
        inference=s['test_policy']+'; '+s['postprocessing'],submission=record['submission'],
        public_score=score,calibration_pq=s['local_calibration_pq'],review_pq=None,score_screenshot=record['score_screenshot'])
d['logs']['snake_public_submission_report']=record['report'];f.write_text(json.dumps(d,ensure_ascii=False,indent=2))
text='\n'.join([
'最佳蛇形模型推理及公开提交（北京时间2026-10-18）','',
f'实际公开分：{score:g}；此前最佳公开分0.36；显示分差：{score-.36:+.8f}。',
'分数以Kaggle页面显示为准，不推断隐藏精度。此次不是本地验证分。',
'模型：原0.36 Mask2Former/R50权重＋动态蛇形共享mask残差模块＋mask_embed delta，选取1200步。',
f'本地校准PQ：{s["local_calibration_pq"]:.8f}。70张图是复用开发集，不能直接换算公开分。',
'完整2048×2048 FP32；原图/水平翻转/垂直翻转/180度旋转四视图；至少两视图IoU>.5一致；均分>=.90；面积>=256；先做贪心不重叠再过滤。',
'此次没有训练、没有阈值搜索、没有评估71图复用验证子集。',
'权重SHA及metadata核对通过；恢复后一张开发图的预测缓存完全匹配此前记录。',
f'处理180张测试图；预测实例{s["predicted_instances"]}个；空预测图{s["empty_test_images"]}张。',
'本地CPU逐图重新构建CSV，验证RLE解码、掩码尺寸/面积、同图不重叠、ID唯一及精确行顺序，PASS。',
f'CSV SHA256：{r["submission_sha256"]}',
f'基础权重SHA256：{s["base_checkpoint_sha256"]}',
f'蛇形delta SHA256：{s["delta_checkpoint_sha256"]}',
f'推理及环境准备墙钟{s["elapsed_seconds"]}秒；仅免费Tesla T4；备份后已Stop session并核验off。',
'GPU页面额度可能滞后，不据此推断精确计费。没有付费、升级、AWS或公开模型/代码。',
'旧权重和旧提交保留。',
f'提交文件：{record["submission"]}',f'评分截图：{record["score_screenshot"]}',f'关闭截图：{record["gpu_off_screenshot"]}',
f'完整执行源码：{O/"executed_complete.ipynb"}',''])
(O/'result_report.txt').write_text(text)
files=[p for p in O.rglob('*') if p.is_file() and p.name!='artifact_sha256.json']+[P/n for n in ['infer_snake_best_body.py','infer_snake_best_run_cell.py','verify_snake_test_submission.py','finalize_snake_test_submission.py']]
(O/'artifact_sha256.json').write_text(json.dumps(dict(created_at_utc=now,sha256={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}),indent=2))
(P/'CONTINUATION_CURRENT.txt').write_text(f'2026-10-18最佳蛇形step1200已推理并提交，公开分{score:g}；GPUoff。\n报告：{O/"result_report.txt"}\nCSV：{record["submission"]}\n当前最佳公开分：{max(.36,score):g}。\n')
print(json.dumps(dict(public_score=score,status=record['status'],best_public_score=max(.36,score)),ensure_ascii=False))
