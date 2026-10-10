# ASR / TTS 下一轮优化空间评估

2026-10-10。当前代码已提交到 `feat/qwen3-audio-rpu`，基线为 **`40e4574`**（ASR 编码器驻留、TTS 输出头输入复用和 CPU codec 转置卷积优化）。本轮只做源码评估和有限局部测量，没有修改或部署新的推理实现。

最值得先验证的是 **ASR CPU 卷积布局**和 **TTS codec 剩余卷积/内存搬运**；随后评估 TTS 真正的 prefill、ASR 输出头融合。TTS 输出头进一步融合需要 native 接口扩展，不能直接开启已有选项。

## 当前正式速度与各阶段空间

完整请求中位数仍为 ASR **294.22 ms**、TTS **3.043 s**。Thor 历史 PyTorch / TensorRT 主体路径分别为 ASR 240.41 / 154.21 ms、TTS 2.574 / 1.057 s；本轮没有重测 Thor。TTS 两机的精度、采样方式、生成步数不同，不能把这些比值作为等工作量的硬件速度比。配置、质量和阶段口径见 [ASR 正式报告](qwen3_asr_resident_r1.zh.md)、[TTS 正式报告](qwen3_tts_chain_optimization_r1.zh.md)及 [Thor TensorRT 报告](qwen3_audio_thor_tensorrt.zh.md)。

以下沿用正式版本的**独立阶段诊断均值**，单位 ms；占比以各自阶段诊断总计计算，不以完整请求中位数计算。

| ASR 阶段 | 当前耗时 | 占比 | 剩余空间与优先级 |
|---|---:|---:|---|
| 特征提取/分词，CPU | 4.08 | 1.4% | 较小，暂缓 |
| 三层卷积，CPU | 107.84 | 37.2% | 首要；本轮布局微测已有收益 |
| 18 层编码器 + 最终 norm + proj1，RPU | 6.35 | 2.2% | 已连续驻留；暂不重写主体 |
| conv_out / proj2，RPU | 1.58 | 0.5% | 可进一步连接边界，但单独收益小 |
| 剩余音频激活/位置/布局，CPU | 17.72 | 6.1% | 先配合卷积布局；改变 GELU 精度有舍入差异 |
| 文本 prefill / decode，RPU | 13.30 / 91.64 | 36.2% | 两者合计已接近 Thor TRT 的 104.73 ms；先优化输出边界 |
| 词表输出头，RPU | 26.58 | 9.2% | 可评估复用已有 fused head；不能直接修改 base model 返回类型 |
| 其他生成/校验/调度 | 20.81 | 7.2% | 分离 argmax、拷贝与调度后再改；保留 finite 校验 |
| 诊断总计 | 289.91 | 100% | 不与完整请求中位数混算 |

| TTS 阶段 | 当前耗时 | 占比 | 剩余空间与优先级 |
|---|---:|---:|---|
| 参考音频读取 | 1.01 | <0.1% | 当前已缓存参考处理，不优先 |
| Talker decoder，RPU | 448.05 | 14.5% | 先验证 prompt 的真正 prefill，减少串行单 token 调用 |
| Predictor decoder，RPU | 1062.67 | 34.4% | 生成主耗时；每帧首两个 token 合并 prefill 是候选，需测 padding 成本 |
| 输出头，RPU | 214.33 | 6.9% | 需要双输出及多 head 选择接口，再考虑融合 |
| 生成组合/调度，CPU | 204.27 | 6.6% | 有大量小调用，但不可把这些时间都视为可删除开销 |
| codec 解码，CPU FP32 | 1156.33 | 37.4% | 首要；继续处理普通卷积、fold 和激活内存遍历 |
| 其他 | 2.03 | 0.1% | 暂缓 |
| 诊断总计 | 3088.69 | 100% | 相同阶段的速度还受生成步数影响 |

## ASR：局部验证发现

在 R1 持设备锁、4 CPU 线程、同一 4.204 秒录音上，提取真实前端输入 `[5,1,128,100]`，连续执行三层卷积、各层 GELU 和末尾布局转换。沿用 FP16 舍入权重、FP32 卷积和 FP16 输出边界，权重预先准备；每方案 2 次预热、5 次计时，不计加载、权重转换和后续 encoder。

| 局部方案 | 中位耗时，ms | 相对原布局 | 最终前端输出 |
|---|---:|---:|---|
| 原 NCHW | 124.37 | — | 参考 |
| 输入及卷积权重采用 channels_last，层间保持布局 | **94.48** | **减少 29.89 ms，24.0%** | 本次样本逐点相同 |
| 原布局，仅第一层 GELU 改为 FP32 计算后转回 FP16 | 112.83 | 减少 11.54 ms | 不完全相同；max-abs 0.0004883 |

这是局部候选结果，**未计入正式 294.22 ms**。局部计时包含 GELU，与上表“纯卷积 107.84 ms”不是同一边界。顺序执行的短测量仍有波动，端到端收益需要单独验证。第一层 GELU 自身的阶段均值仅从 15.14 降至 11.21 ms，其余表面收益含卷积波动，不能把 11.54 ms 全归因于 GELU，也不能与布局收益相加。

最小改动位置是 [ASR `_install_rpu`](../python/rpu_backend/api/qwen3_asr.py)：安装卷积时一次性准备 channels_last FP32 权重，输入转换后保持该布局。激活、模型精度、RPU 路由和 shape 契约均可沿用。验证应先检查输出形状/有限值及原有 host checks，再做 2+5 完整请求、已有 20 条识别样本和长度切换/Graph 生命周期检查。

此前现成 RPU 卷积接口三层合计 392.69 ms，CPU 对应 94.94 ms；受 batch/通道补齐和边界转换影响，直接替换反而更慢。后续若考虑 RPU 卷积，必须先找到避免过量补齐、能连续接入后续 encoder 的路径，不能重复同一独立调用方案。

ASR 词表输出头的第二优先级机会：

- `_generate` 每步先获得 decoder hidden，再单独调用 `_project_logits`，随后把 logits 复制到 CPU 做 finite 检查和 argmax。
- [Qwen3 adapter](../python/rpu_backend/adapters/qwen3.py) 已有 `_apply_fused_lm_head_for_rpu` 和 fused decode top1；[native `set_lm_head`](../src/fused/rpu_qwen3_model.h) 的 FP16 decode head 能接受此 hidden/vocab 几何。保留原 FP32 累加配置，不需要另写矩阵乘算子。
- 但 [decoder runtime](../python/rpu_backend/runtime/decoder.py) 的 `_validate_decoder_base_output` 要求返回 hidden。ASR 当前装的是 base model，不能直接给同一 handle 设置 head 后继续使用原返回契约；需要在 owner/API 层复用 causal-LM 路径，保留 prefill 最后位置的独立投影及关闭/Graph 管理。
- 现有 host top1 是 CPU 直接扫描 RPU DDR logits，并非 RPU 内部 argmax；它也不等价于现有非有限值拒绝。不得为加速删除 finite 检查。
- 26.58 ms 包括词表权重读取和真实计算，可优化的是独立提交、输出搬运及取最大值的部分，不能承诺整段消除。本轮没有测量 fused head 候选。

新长度请求的 BUILD 延迟另需关注：正式固定输入首请求约 771 ms，稳态约 294 ms；20 条不同长度录音的均值仍为中文约 730 ms、英文约 828 ms。当前按长度清图是为控制 launch arena。长度分桶或多图复用必须同时核实有效长度、mask、音频占位符、KV 位置与内存预算，不能简单放宽缓存上限。

## TTS：codec 剩余耗时的实际构成

沿用冻结最终 codec、固定 31 帧 codes、8 CPU 线程，持锁预热 2 次后做 **1 次 CPU profiler**。输出 59520 个采样点，与已有最终波形逐点一致。含插桩总时为 1134.87 ms，仅用于归因，不替换正式 codec 或完整请求速度。

以下 leaf module 计时互不嵌套；“剩余”是总时减去 leaf 时间，包含非 leaf 运算及插桩，不能全部理解为 Python 调度。

| 构成 | 耗时，ms | 后续方向 |
|---|---:|---|
| 普通 Conv1d | **546.39** | 其中 kernel=7 为 409.24 ms，kernel=1 为 135.95 ms；优先针对真实形状评估 FP32 预打包/布局 |
| 已改为 matmul + fold 的 ConvTranspose1d | **212.14** | 重点减少 GEMM 输出布局转换与 fold 中间数组 |
| SnakeBeta 激活 | **129.98** | 评估合并逐元素内存遍历；保留 sin、平方、缩放和残差语义 |
| Linear | **115.44** | 位于 codec 前段；先看同一块连续执行机会，避免逐层上传/下载 |
| 其他 leaf modules | 9.80 | 非主要瓶颈 |
| 非 leaf 运算与插桩等剩余 | 121.12 | 如需继续优化，应进一步归因 |
| profiler 总计 | 1134.87 | 单次诊断 |

另一视角的 ATen 算子时间：`copy_` self 142.82 ms、`col2im` self 72.99 ms（含子调用 total 104.38 ms）、`sin` self 57.44 ms。它们已经包含在上表模块时间中，**不能再相加**，也不能把全部 copy 时间归给转置卷积。

[当前 codec 实现](../python/rpu_backend/adapters/qwen3_tts/codec.py) 的转置卷积为 `matmul → transpose → fold → bias`。四个主要上采样层满足 kernel=2×stride，可试相邻块重叠相加；另外两个前段上采样层为 kernel=stride=2，没有同样的重叠。应先在临时函数验证 FP32 累加、输出长度、bias 和外层因果裁剪，再决定是否替换。若带来新的大临时数组，可能抵消算子节省。

普通 Conv1d 全量 channels_last 替换此前已试过：固定输入 codec 从 1.150 s 变成 1.172 s，没有收益，因此本轮没有重复。下一步只能针对热点形状局部试验，并用**完整 codec**判断是否值得保留；ASR Conv2d 的布局收益不能直接外推到 TTS Conv1d。

codec 放到 RPU 仍有较大接口工作：需要核对带 dilation 的一维卷积、上采样重叠相加、SnakeBeta 和连续内存布局。先用一段真实 residual/upsample block 评估计算与边界传输，证明连续执行有收益后再扩展；逐个 Linear 搬上 RPU 不能证明整体会快。codec 当前是 FP32，改 FP16 也会成为新的精度配置，不能只凭单层误差小就部署。

## TTS：decoder、输出头与调度

[TTS `_decoder_forward`](../python/rpu_backend/api/qwen3_tts.py) 当前将所有输入 token 拆成单 token Graph replay，每次做 CPU FP32 校验、FP16 转换、上传、RPU decoder、下载和输出校验。

1. **真正 prefill**：固定样本 Talker prompt 为 18 token，合并可在调用层减少额外 17 次单 token 执行。Predictor 每帧首两个 token 也可评估合并。[现有 decoder runtime](../python/rpu_backend/runtime/decoder.py) 有 prefill 规划可复用；但需要独立的 prefill descriptor/Graph 签名、位置和 KV 更新，以及多 token hidden 输出契约。短序列 prefill 可能因对齐反而更慢，减少调用次数不是已证明的延迟收益。
2. **输出头融合**：固定样本共 497 次 head 调用，其中 Predictor 为 465 次、使用 15 个不同 head。Talker 的 normalized hidden 仍供下一帧 Predictor 使用；普通 native fused head 仅输出 logits，同时返回 hidden 的 exact-rerank 路径仅适用于 Qwen3-VL-2B W4。需要显式双输出契约和独立输出存储；Predictor 则需一次安装 head bank、以受控 index 选择，不能每步调用 setter 触发 `invalidate_model_state()`。index 必须进入图语义或通过已有约定的 mutable DMA 更新。
3. **调度**：现成 RhinoVLA 风格 fast replay 通用分支有 position=0、seq>1 等限制，不能给逐 token TTS decode 简单加 flag。优先减少可合并的 prefill 边界；之后再测 position/mask 分配、CPU 转换和校验占比，保留请求输入校验和错误处理。

输出头和调度收益可能重叠；不能把当前 214.33 ms 与 204.27 ms 直接作为可省时间相加。参考音频读取约 1 ms、当前热路径已命中参考缓存，不优先扩大缓存策略。

## 实施顺序与验收

| 顺序 | 候选 | 最小验证 |
|---|---|---|
| 1 | ASR CPU channels_last 卷积 | 本轮局部结果已过；下一步 host checks、2+5 完整请求、20 条 CER/WER/token 对照及已有长度/Graph 检查 |
| 2 | TTS codec 热点 Conv1d、转置卷积布局/重叠相加 | 真实形状局部数值与边界检查；固定 codes 完整 codec 2+5；收益成立后再做完整请求及既有 10 条独立转写 |
| 3 | TTS Talker / Predictor 真正 prefill，分别评估 | KV/位置及所有返回 hidden 对照；一次 BUILD 后稳定 REPLAY；相同生成长度完整请求计时 |
| 4 | ASR 复用 fused head | 保留 FP32 累加、finite 拒绝、prefill/decode 输出契约；token、Graph 与完整请求验证 |
| 5 | TTS 双输出及多 head native 接口；codec RPU 连续块 | 先单块/单组件硬件验证；有净收益再推进完整链路与任务质量 |

顺序按现有证据、改动范围与风险综合确定，不是对收益大小的承诺。本轮没有新增测试门槛或扩大输入支持范围。TTS 转写只检查文字内容，若改变 codec 精度或近似激活，还必须单独评估音质，不能拿 CER/WER 代替。

[机器可读摘要](evidence/qwen3_audio_next_optimization/summary.json)保存 ASR 全部局部计时及 codec 聚合归因，不包含 codes、波形、中间激活或底层 trace。ASR 复现脚本在 R1 的 `/home/agi/rhinoforge_audio_runtime/opportunity_20261010/asr_frontend_probe.py`；TTS profiler 使用原 `tts_chain_20261010` 冻结 codec 和固定 codes。正式版本的质量证据继续引用上述已有报告，本轮未做新候选的端到端质量认证。
