# Solar M2F · snake2048

Instance segmentation experiments on solar imagery, built on Mask2Former R50 at native 2048×2048.

Best public leaderboard score: **0.36** (step1200 submission, 2026-10-18).

---

## 我的角色 / My role

I owned this line of work end to end. A teammate contributed one early Mask2Former
baseline at 512×512; everything after that — the snake adapter, the native-2048
finetune, the four-view consensus inference, the submission and its verification —
is mine. I am not claiming the initial baseline.

我负责这条线的全部后续工作。队友只在最开始提供过一版 512×512 的 Mask2Former baseline，
之后的蛇形模块、2048 原生微调、四视图一致性推理、提交与校验都是我自己完成的。

## 这个项目在做什么 / What this is

Solar panels are hard to segment at full resolution: a 2048×2048 image contains
many small, partly-occluded instances, and a model trained at 512×512 loses exactly
the detail that matters. The work here is about recovering that detail without
retraining a full model from scratch.

太阳电站图像在高分辨率下实例分割很难：一张 2048×2048 图里有大量小尺寸、互相遮挡的目标，
512×512 训练的模型恰好会丢掉最关键的细节。这套工作解决的就是如何在不从头重训的前提下把细节找回来。

## 三段主要工作 / The three main pieces

**1. 动态蛇形共享掩码残差模块 / Dynamic snake-shaped shared-mask residual**

An adapter inspired by DSConv: a shared mask, modulated by `mask_embed`, predicts a
DSConv-style offset field that deforms the mask feature map. The base model stays
frozen — only `mask_embed` and the adapter train. This is what lets a 512-trained
backbone produce masks that follow snake-like boundaries at 2048.

参照 DSConv 的适配器：用 `mask_embed` 调制一张共享掩码，预测 DSConv 风格偏移场来形变掩码特征图。
基础模型全程冻结，只训练 `mask_embed` 和适配器本身。这正是让 512 训练的骨干网络能在 2048 分辨率上
贴合蛇形边界的关键。

**2. 2048×2048 原生分辨率微调 / Native-2048 finetune**

Matched finetune at 2048 against a frozen base. Selected at step1200.

针对冻结基础模型的 2048 原生分辨率配对微调，最终选用 1200 步的检查点。

**3. 四视图一致性推理 / Four-view consensus inference**

Full FP32 inference at 2048 over the original image plus horizontal flip, vertical
flip, and 180° rotation. An instance is accepted when at least two views agree with
IoU > 0.5, mean score ≥ 0.90, and area ≥ 256; greedy non-overlap runs before the
filter. This costs four forward passes and removes most of the fragmentation that
single-view inference produces on thin structures.

在 2048 上做全 FP32 推理，对原图、水平翻转、垂直翻转和 180 度旋转四个视图分别预测。
一个实例至少要有两个视图 IoU > 0.5、平均分 ≥ 0.90、面积 ≥ 256 才被保留，再跑一次贪心去重叠后过滤。
代价是四次前向，换掉的是单视图在细长结构上最严重的碎片化问题。

## 结果 / Results

| 指标 | 值 |
| --- | --- |
| 公开榜分 / Public LB | **0.36** |
| 本地校准 PQ / Local calibration PQ | 0.46036319（baseline 0.45637） |
| 测试图 / Test images | 180 |
| 预测实例 / Predicted instances | 1068 |
| 空预测图 / Empty predictions | 6 |
| 推理墙钟 / Inference wall clock | 1571 s（免费 Tesla T4） |

Note the 70 calibration images are reused from the development set, so the local PQ
cannot be converted into a leaderboard estimate. The 0.36 is the number Kaggle
displayed; no hidden-set metric is inferred.

那 70 张校准图是从开发集复用的，所以本地 PQ 不能换算成榜分。0.36 是 Kaggle 页面显示的公开分，
没有对隐藏集做任何推断。

## 可复现性 / Reproducibility

Every artifact is hash-pinned — base weights, adapter delta, and the submission CSV:

```
CSV SHA256      af491b65a31f870278c4a67a2b0331fd0f24b951f44d9aff1bdda1b1f5d3270b
base weights    6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1
snake delta     ca87077f721356018594b3f0fe4cf3bea562991c5f3e484347782cfa7946ad75
```

The submission CSV was rebuilt on CPU image by image and re-validated for RLE decoding,
mask size and area, non-overlap within an image, ID uniqueness, and exact row order —
that check passes (`evidence/result_report.txt`).

提交 CSV 是在 CPU 上逐图重新构建的，并重新校验了 RLE 解码、掩码尺寸与面积、同图不重叠、
ID 唯一性和精确行顺序，全部通过。

All compute ran on a free Kaggle Tesla T4. No paid tier, no cloud credits, no
private model or code was used.

所有计算都跑在 Kaggle 免费的 Tesla T4 上，没有使用付费档位、云额度或外部私有模型与代码。

## 目录结构 / Layout

```
src/         46 个 Python 文件，约 6380 行 / 46 Python files, ~6380 lines
notebooks/   推理与微调 notebook / inference and finetune notebooks
notes/       过程记录：策略、对齐、实验报告 / strategy, alignment and experiment notes
evidence/    提交凭证、哈希、公开分截图 / submission receipt, hashes, score screenshot
```

## 目录内不包含的东西 / Deliberately excluded

Model checkpoints (`.pth`), result archives, vendored dependencies, and virtual
environments are not committed. The submission description and the provenance record
in `evidence/` carry the exact configuration needed to reproduce the numbers.

模型权重、结果压缩包、第三方依赖和虚拟环境都没有提交。`evidence/` 里的提交描述和
provenance 记录保留了复现这些数字所需的完整配置。
