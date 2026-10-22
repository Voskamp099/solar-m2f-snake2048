from pathlib import Path
import json,hashlib,datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;O=P/'snake2048_controls_20261018'
s=json.loads((O/'summary.json').read_text());v=json.loads((O/'independent_verification.json').read_text());assert v['status']=='PASS'
labels={'head_only':'只微调mask_embed','fixed_snake':'mask_embed＋固定条带分支','snake':'mask_embed＋动态蛇形分支'}
lines=['2048完整图像：Mask2Former共享mask分支三组对照','日期：2026-10-18（北京时间）','',
'范围：用户授权的免费GPU三组小试。未修改/替换公开0.36权重，未生成或提交CSV，未评估71图复用验证子集。',
f"训练输入：完整2048×2048，无裁剪、无缩小。每组{s['effective_updates_per_arm']}次有效更新。",
'原模型只解冻sem_seg_head.predictor.mask_embed；分支组额外训练共享mask_features上的256→32→256残差适配器，不修改其他尺度或各decoder层。',
'固定条带与动态蛇形使用相同初始化及结构；固定组偏移恒零，动态组允许学习累计有界偏移。',
'学习率：mask_embed1e-6；分支1e-4；残差门控1e-3。AdamW，weight_decay=.01，梯度裁剪1，AMP FP16。',
'数据：原566张训练图，逐物理图均匀采样，再随机选标注；三组完全相同的图像/标注/水平垂直翻转顺序。',
'推理：完整2048 FP32四视图；固定2视图共识IoU>.5、平均分数>=.90、面积>=256、贪心去重。',
'', '显存预检（包含反向与Adam状态，最多24实例的训练标注）：']
for x in s['preflights']:lines.append(f"{labels[x['arm']]}：{x['status']}，峰值allocated={x['peak_allocated_gib']:.4f}GiB，reserved={x['peak_reserved_gib']:.4f}GiB；2个优化器预检步骤丢弃后重新加载原权重。")
lines+=['','校准PQ，不是Kaggle公开分：',f"原权重基线：{s['baseline_calibration_pq']:.8f}"]
for a in s['arms']:
 m=a['calibration'];c=v['results'][a['arm']]['diagnostic_counts'];lines.append(f"{labels[a['arm']]}：PQ={m['PQ']:.8f} SQ={m['SQ']:.8f} RQ={m['RQ']:.8f}；TP={m['tp']} FP={m['fp']} FN={m['fn']}；更新{a['effective_updates']}/{a['attempts']}次尝试；训练峰值{a['training_peak_allocated_gib']:.3f}GiB；该组训练加评估{a['elapsed_seconds']}秒。")
 lines.append(f"  小目标匹配{c['small_gt_matched']}/{c['small_gt']}（标注面积<2000像素）；一个GT重叠多个预测={c['gt_overlapping_multiple_predictions']}；一个预测重叠多个GT={c['predictions_overlapping_multiple_gt']}。")
b=v['results']['baseline']['diagnostic_counts'];lines+= [f"基线小目标匹配{b['small_gt_matched']}/{b['small_gt']}；一个GT重叠多个预测={b['gt_overlapping_multiple_predictions']}；一个预测重叠多个GT={b['predictions_overlapping_multiple_gt']}。",'重叠计数使用IoU>0，仅为分裂/合并代理，不直接当作物理错误判决。IoU/Dice分布另存于independent_verification.json。']
best=max(s['arms'],key=lambda a:a['calibration']['PQ']);gain=best['calibration']['PQ']-s['baseline_calibration_pq']
lines+=['',f"本轮最佳：{labels[best['arm']]}，相对原基线校准PQ差值{gain:+.8f}。",'判断仅限本轮校准证据，单种子、短训和已复用开发数据不足以确定充分训练效果、架构上限或独立测试泛化。',
'预算：计划最多600次有效更新，按预检最慢样本速度及保守双GPU额度预算统一收缩到250；这不是训练已经收敛。三组检查点均支持继续原采样序列。',
f"实验主体墙钟：{s['elapsed_seconds']}秒，另有环境安装和会话启动开销。账户显示可能滞后，不把显示读数差当作精确计费。",'核验：本地CPU独立复算全部PQ/SQ/RQ及TP/FP/FN一致；三组训练序列一致；原模型冻结部分运行时逐张量检查一致；mask_embed确有更新，固定偏移保持零、动态偏移确有更新；metadata、优化器、scaler及RNG均已保存。',
'论文表述：这是DSConv启发的共享mask残差适配，不是完整DSCNet复现；已有DSConv+Mask2Former先例，不宣称首次组合。',
'', '文件：summary.json、events.jsonl、*_history.json、*_candidates.json、*_last.pth、training_schedule.json、independent_verification.json、实际GPU输出live_gpu_stdout.txt。',
'恢复：先加载原0.36基线（SHA256='+s['source_sha256']+'），按adapter_config挂接分支，再加载delta并严格检查允许缺失键；恢复优化器/scaler/RNG，并从training_schedule[effective_updates:]继续。',
'完整原权重位于 '+str(P/'finetune_results/model_best_finetune.pth'),
'本轮GPU关闭证据：'+str(O/'gpu_off.png')]
(P/'snake2048_controls_20261018_report.txt').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
vals=[s['baseline_calibration_pq']]+[a['calibration']['PQ'] for a in s['arms']];names=['Original','Head only','Fixed strip','Dynamic snake']
axes[0].bar(names,vals,color=['#888888','#849e89','#a8b89c','#296b50']);axes[0].set_ylabel('Calibration PQ (not Kaggle public score)');axes[0].set_ylim(0,max(vals)+.04)
for i,x in enumerate(vals):axes[0].text(i,x+.003,f'{x:.5f}',ha='center',fontsize=9)
for a in s['arms']:
 h=json.loads((O/(a['arm']+'_history.json')).read_text());y=np.array([r['loss'] for r in h if r['optimizer_updated']]);window=25
 axes[1].plot(np.arange(window,len(y)+1),np.convolve(y,np.ones(window)/window,mode='valid'),label=a['arm'])
axes[1].set_xlabel('Effective optimizer updates');axes[1].set_ylabel('Training loss, 25-update rolling mean');axes[1].legend(fontsize=8)
fig.suptitle(f"Native 2048 matched controls | one seed | {s['effective_updates_per_arm']} updates per arm")
fig.savefig(O/'comparison.png',dpi=170);plt.close(fig)
print('REPORT_AND_COMPARISON_WRITTEN')
