# EEG Router

**癫痫检测中的事件路由与误报评估代码。**

[English](README.md) · [实验索引](docs/EXPERIMENTS.md) · [评估协议](docs/PROTOCOL.md) · [外部数据要求](docs/DATA.md) · [实验状态](docs/STATUS.md)

作者：Ziquan Wang、Zixuan Luo（共同第一作者）。

## 研究问题与代码范围

在未参与训练的患者上，从已有 EEG detector 的候选报警事件中进行选择，能否降低误报？本仓库整理了论文 *Oracle Decomposition of False-Alarm Control in Routed EEG Seizure Detection* 及后续分析代码。

包含 SingLEM、CBraMod 和 TCN-GAT 的训练/推理支持、事件 router、同候选组 oracle、误报时间共享、detector 来源分类，以及后续对照分析。原 router 的阴性结果保留。原论文、替代协议和开发中的补充分析分别列在[实验索引](docs/EXPERIMENTS.md)，不能混用结果。

## 无需 EEG、GPU 或权重的快速示例

使用 Python 3.11：

```bash
git clone https://github.com/dennis216/eeg-router.git
cd eeg-router
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python examples/evaluation_demo.py
```

Windows PowerShell 将激活命令替换为 `.venv\Scripts\Activate.ps1`。通过绝对路径调用示例时，也可以从其他目录运行。

[示例](examples/evaluation_demo.py)使用固定合成分数和 seizure 区间，调用现有事件化、匹配和指标函数。预期 TP=1、FP=1、敏感度=0.5、FA/24h≈24.1341。分母为一小时合成记录中的 3,580 秒背景时间。这些数字仅用于教学，不是论文结果，也不代表完整 router 复现。

运行轻量测试：

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/evaluation tests/examples -q
python scripts/check_public_repository.py
```

## 进入正式实验

先阅读[实验索引](docs/EXPERIMENTS.md)、[原始患者划分规范](protocols/v3-groupkfold-confirmatory-v1.md)和[实际评估协议](docs/PROTOCOL.md)。原协议采用患者级五折、三个种子（17、42、3407）。

正式分析需要额外依赖、数据、保存的预测和 validation 锁等[外部输入](docs/DATA.md)。部分历史脚本使用 `/root/b_false_alarm_atlas` 等绝对路径，应逐项检查，不能覆盖已有实验目录。仓库目前不承诺完整实验一键复现。

## 实验状态

截至 2026-09-30 的已核验清单，评审补充分析只有单 run smoke、bootstrap=0；完整 15-run 分析尚未核验完成。[状态说明](docs/STATUS.md)记录待解决问题。本次文档、示例和自动测试不会改变实验完成状态。

## 分享边界

公开版不含患者数据、权重、凭据、评审邮件或论文稿件。完整历史备份保持私有。当前未为作者代码指定开源许可证，公开可见不等于授予一般复用许可；第三方代码和数据须遵守各自条款。
