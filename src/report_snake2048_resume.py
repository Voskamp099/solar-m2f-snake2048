from pathlib import Path
import json,datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;O=P/'snake2048_resume_20261018'
s=json.loads((O/'summary.json').read_text());v=json.loads((O/'independent_verification.json').read_text());r=json.loads((O/'session_receipt.json').read_text())
assert v['status']=='PASS' and r['session_off']
labels={'head_only':'只微调mask_embed','fixed_snake':'固定条带＋mask_embed','snake':'动态蛇形＋mask_embed'}
lines=['完整2048三组匹配续训记录（2026-10-18）','',
'先澄清提交状态：本次续训没有提交新CSV。0.36是此前原模型已经提交得到的公开分，不是本次蛇形模型提交后的分数。下文所有PQ都是70张复用开发图上的本地验证分，不能直接预测Kaggle公开分。','',
'用户授权继续使用剩余免费GPU。启动前页面显示01:51/30h已用，剩余约28h9m；本轮只用免费T4，禁止付费、AWS和公开发布。',
'三组从各自250步检查点恢复，保留原优化器/scaler/CPU及CUDA随机状态，严格延续同一图像/标注/翻转序列。恢复后的预测缓存匹配，恢复张量逐项一致；预检更新丢弃后再恢复原检查点。',
'输入完整2048×2048，不裁剪、不缩小。冻结原模型其他部分，只训练mask_embed及各组新增共享mask残差模块。',
'超参数与250步试验一致：mask_embed LR1e-6，分支LR1e-4，门控LR1e-3，AdamW weight_decay=.01，梯度裁剪1，AMP FP16。',
'预先规定：三组都先到600；若至少一组600步PQ比该组250步高>.001，则三组都续到1200，否则停止。0.001仅控制是否继续的资源规则，不是统计显著性门槛。','',
'250步本地验证结果：']
for mode,q in s['parent_calibration_pq'].items():lines.append(f'{labels[mode]}：{q:.8f}')
lines+=['',f"原0.36模型本地验证基线：{s['baseline_calibration_pq']:.8f}",'续训本地验证结果：']
all_results=[]
for phase in s['rounds']:
 lines.append(f"阶段：{phase['target']}次有效更新")
 for a in phase['arms']:
  all_results.append(a);m=a['calibration'];label=a['arm']+'_step'+str(phase['target']);c=v['results'][label]['diagnostic_counts']
  lines.append(f"  {labels[a['arm']]}：PQ={m['PQ']:.8f} SQ={m['SQ']:.8f} RQ={m['RQ']:.8f} TP={m['tp']} FP={m['fp']} FN={m['fn']}；有效更新/累计尝试={a['effective_updates']}/{a['attempts']}；该段训练及评估{a['elapsed_seconds']}秒。")
  lines.append(f"    小目标匹配={c['small_gt_matched']}/{c['small_gt']}；一个GT重叠多个预测={c['gt_overlapping_multiple_predictions']}；一个预测重叠多个GT={c['predictions_overlapping_multiple_gt']}。")
best=max(all_results,key=lambda a:a['calibration']['PQ']);gain=best['calibration']['PQ']-s['baseline_calibration_pq']
lines+=['',f"本轮最好：{labels[best['arm']]}，{best['effective_updates']}步，PQ={best['calibration']['PQ']:.8f}，相对原模型差值{gain:+.8f}。",f"停止状态：{s['status']}；{s.get('stop_reason','预设阶段完成。')}",
'小目标以标注面积<2000像素定义，是面积代理，不等同于所有细暗条；多重重叠按IoU>0计数，只是分裂/合并代理，不能直接当作物理错误判决。',
'核验：本地CPU重新计算新候选的PQ/SQ/RQ及TP/FP/FN，与GPU结果一致；已验证未修改的旧候选按SHA256复用其独立CPU结果。每个阶段三组采样序列匹配，250步历史前缀保持原样，metadata版本2、优化器/scaler与RNG完整。',
'局限：单种子、已复用开发集、冻结大部分模型；不能据此断言架构上限、独立测试泛化、获奖或论文接收。没有用71图复用验证子集反复调参。',
'没有修改既有0.36原权重，没有生成或提交新CSV，没有公开模型/仓库。',
f"实验主体墙钟{s['elapsed_seconds']}秒，另有环境安装和会话启动/备份开销。结束后GPU已关闭；页面额度显示{r.get('quota_display','未重新读取')}，不把滞后读数当精确计费。",'',
'恢复：加载原0.36权重，核对源SHA，再按adapter_config挂接对应分支并加载delta；恢复优化器/scaler/RNG，从training_schedule[effective_updates:]继续。',
'原权重：'+str(P/'finetune_results/model_best_finetune.pth'),
'日志与检查点目录：'+str(O),
'完整执行Notebook：'+str(P/'snake2048_resume_executed_complete_20261018.ipynb'),
'GPU关闭证据：'+str(O/'gpu_off.png')]
(P/'snake2048_resume_20261018_report.txt').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for mode in ('head_only','fixed_snake','snake'):
 xs=[250];ys=[s['parent_calibration_pq'][mode]]
 for phase in s['rounds']:
  a=next(a for a in phase['arms'] if a['arm']==mode);xs.append(phase['target']);ys.append(a['calibration']['PQ'])
 axes[0].plot(xs,ys,'o-',label=mode)
 last=s['rounds'][-1]['target'];h=json.loads((O/(mode+'_step'+str(last)+'_history.json')).read_text());y=np.asarray([row['loss'] for row in h if row['optimizer_updated']]);window=50
 axes[1].plot(np.arange(window,len(y)+1),np.convolve(y,np.ones(window)/window,'valid'),label=mode,alpha=.85)
axes[0].axhline(s['baseline_calibration_pq'],ls='--',c='gray',label='Original baseline');axes[0].set_xlabel('Effective updates');axes[0].set_ylabel('Local validation PQ (not Kaggle public score)');axes[0].legend(fontsize=8)
axes[1].set_xlabel('Effective updates');axes[1].set_ylabel('Training loss, 50-update rolling mean');axes[1].legend(fontsize=8)
fig.suptitle('Native2048 matched continuation | one seed | resumed optimizer and RNG')
fig.savefig(O/'comparison.png',dpi=170);plt.close(fig);print('CONTINUATION_REPORT_AND_PLOT_WRITTEN')
