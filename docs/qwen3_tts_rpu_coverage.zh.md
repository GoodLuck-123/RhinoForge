# Qwen3-TTS 0.6B Base 的 RPU 覆盖与剩余 CPU 路径

本评估针对 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，检查点 revision
`5d83992436eae1d760afd27aff78a71d676296fc`，已审核上游 revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e`。起点为 RhinoForge `f8f0d9f`。
R1 使用 RhinoForge-runtime-v1.1.0 资产包、Launch 1.0.0 commit `613f2da`。
沿用单请求、greedy、非流式、文本至多 32 个 tokenizer token、生成至多 64 步的公开范围。

目标是增加**实际由 RPU 算子执行的计算**。张量的 `device` 为 `rpu`、能返回结果，
均不能证明计算发生在 RPU：多个公开算子明确使用 CPU fallback。
保留现有 codec 的 CPU FP32 优化，不把转换为 FP16 称作等价迁移。

## f8f0d9f 起点执行位置

|部分|当前精度和位置|当前可以直接复用的能力|
|---|---|---|
|Talker、Predictor Transformer|RPU FP16，含 attention/norm/SiLU/Linear|现有 fused causal decoder；Talker 有真正 prefill，Predictor 保持 decode|
|16 个输出头|RPU FP16|单 token 直接消费当前 decoder 的 RPU hidden|
|文本/音频码 embedding、生成组合|CPU FP32|标准 `aten::embedding` 在 RPU 上仍调用 CPU；专用 gather 只接受 FP16|
|codec 码本反量化|CPU FP32|先除以 cluster usage，再查表、逐码本求和及投影；当前没有整段 FP32 RPU 实现|
|codec 前段 8 层 Transformer|CPU FP32|共享 decoder 有部分匹配结构，独立 FP16 候选见下文；不是已认证 profile|
|codec 卷积、上采样、SnakeBeta|CPU FP32|保留转置卷积矩阵乘与相邻块叠加、1×1 卷积矩阵乘|
|参考音频、speaker encoder|CPU FP32|同参考音频复用已计算的 speaker embedding；涉及 STFT、mel 和 ECAPA-TDNN|

此前无编译争用的正式阶段测量：Talker 298.36 ms、Predictor 1058.20 ms、输出头
206.83 ms、codec 1015.37 ms。它们包含各自的传输和调度，不能当作设备纯计算占用率。
见[第二轮优化报告](qwen3_audio_optimization_v2_r1.zh.md)。

## FP32 codec 不能直接迁移的源码证据

|需要的运算|当前公开路径|结论|
|Conv/ConvTranspose|`src/ops/rpu_conv2d.cpp` 的 `rpu_conv2d` 对 `transposed` 或非 Half 输入直接跳到 `cpu_fallback`，以 CPU Float 卷积计算后复制回输入设备|FP32 或转置卷积没有原生路径|
|Linear|`src/ops/rpu_linear.cpp` 的 `rpu_linear` 对非 Half 调用 `cpu_fallback_linear`；`linear_with_accumulation` 要求 Half 输入、Half/W8/W4 权重、Half 输出|FP32 accumulation 不等于 FP32 输入/输出支持|
|普通 `matmul`|`src/core/rpu_tensor_ops.inc` 的 `rpu_matmul` 无条件使用 CPU `at::matmul`，随后复制到 RPU 输出|不能把现有 codec 的矩阵乘 helper 直接 `.to('rpu')` 后称为 RPU GEMM|
|SnakeBeta 的 sin|同文件 `rpu_sin` 对非 Half 使用 CPU `at::sin_out`|FP32 sin 没有原生路径|
|SnakeBeta 的乘、加、除|`src/ops/rpu_eltwise_binary.cpp` 按 promoted dtype 检查，非 Half 转 CPU 运算|SnakeBeta 不能在保持 FP32 时连接成 RPU 连续段|
|RMSNorm|`src/ops/rpu_rmsnorm.cpp` 的 `rpu_rmsnorm` 明确只接受 FP16 输入和权重|FP32 norm 没有对应公开原生接口|
|embedding|`src/core/rpu_dispatch_registrations.inc` 的 `aten::embedding` 注册显式 CPU fallback；`src/ops/rpu_embedding.cpp` 的 `gather_embedding` 要求 `[V,H]` FP16、int32 ids、H 为 16 的倍数|现成 helper 无法保持当前 FP32 表的数值精度|
|GELU|评估起点的 `aten::gelu` 注册为 CPU fallback；fused host launcher 已有明确 ERF/TANH 模式，但 SPM 输入输出仍为 FP16|可为已用 FP16 的边界增加薄封装；不能据此宣称 FP32 ConvNeXt GELU 已支持|

FP32 内存存取或 DMA 字节拷贝能力不能替代 FP32 数值算子。此次没有使用位重解释、
隐式降精度或把 CPU fallback 伪装为设备执行的方式。

## 即使改成 FP16，卷积仍有几何限制

1D 卷积需要先按正确布局映射到 `[N,C,1,T]` 的 2D 包装。现有单核路径要求
输入通道对齐，group 路径还要求每组输入通道为 16 的倍数；ConvNeXt 的
`groups=1024、Cin/groups=1` 不满足。多核路径只支持 `groups=1`，并要求
`N % num_cores == 0`；公开语音请求 `N=1`，不能用这一路径自动分摊长时序。
依据为 `src/ops/rpu_conv2d.cpp` 的单核/多核分支和 `rpu_conv2d_fits_in_spm`。

用已测 31 帧、2.48 秒语音的真实张量形状，按该函数原式计算单核 SPM 需求：

|普通 K7 卷积输入 `[N,C,T]`|输出通道|单核 SPM 需求|
|---|---:|---:|
|`[1,768,998]`|768|11.02 MiB|
|`[1,384,4966]`|384|9.62 MiB|
|`[1,192,19846]`|192|16.28 MiB|
|`[1,96,59526]`|96|25.57 MiB|

这些已超过包装函数使用的 8 MiB 物理容量，框架保留开销后的可规划空间只会更小。
因此“只把卷积转 Half”仍不能让这些真实几何走当前 RPU 路径。
这里是源码容量公式的计算结果，不是假称已经在板上测得的耗时。

要保留当前 FP32 codec，缺口包括：FP32 GEMM/Conv1d（含 dilation 1/3/9 与 depthwise）、
长时序的分块及 halo 处理、转置卷积/重叠叠加、FP32 norm 与 sin/逐元素运算，
以及具备明确 FP32 表输入契约的 gather。需要设备资产与 host 封装共同支持，
不是修改模型配置或添加一个设备标记可以解决。

## 独立 FP16 Transformer 前段候选

codec 前段为 8 层，hidden 512、intermediate 1024、16 个 Q/KV head、head_dim 64，
SiLU gated MLP；Q/K norm 为 Identity。共享 causal decoder 的空 Q/K norm 列表
可以表达 Identity，投影宽度与 hidden 不相等也有显式支持。

仍有两个必须明确的新条件：

- 上游每层 attention/MLP 残差前有 learned LayerScale；现成 decoder 无此独立参数。
  候选在 FP32 中将 scale 分别折入 `o_proj`、`down_proj` 的输出行，然后转换 Half。
  对应 CPU 参考必须使用同一折叠配方；与原 FP32 结果的差异另行记录。
- 上游 sliding window 为 72；仅在本次 `1–63` 个 codec 帧的范围内，滑窗才与普通
  causal attention 相同。不能将该候选推广到长于 72 帧或通用 codec 分块解码。

现有公开 Qwen3/Llama envelope 没有该 8 层/512 几何。隔离探测只检查真实 31 帧片段，
以现成 native decoder 加 RPU 输入/输出投影构成 FP16 候选，Linear 使用 FP32 accumulation。
这属于独立探测阶段，不构成整个 codec 的 FP16 支持声明；最终 API 集成结果见下文。

在真实 31 帧、59520 个采样点（2.48 秒）的固定 codec 输入上，8 CPU 线程，
共用 RPU 锁排除其他模型与编译，2 次预热后测 5 次，中位数如下：

|阶段|CPU FP32 基线|RPU FP16 前段候选|变化|
|---|---:|---:|---:|
|前段 Transformer，含输入/输出投影及传输|112.65 ms|3.07 ms|−97.27%|
|完整 codec，其他阶段保持 CPU FP32|1081.36 ms|966.81 ms|−10.59%|

匹配 CPU FP16 折叠参考的前段耗时为 98.27 ms。RPU 对该参考的 MSE 为
`8.96e−10`、max-abs 为 `0.00024414`、相对 L2 为 `0.1666%`；
对原 CPU FP32 前段的相对 L2 为 `0.1750%`。输出有限、形状正确，7 次输出逐位一致，
Graph 保持 1 项、6 次 replay，invariant 通过。这些是数值观察，不是替代任务质量的门槛。

完整波形保持 59520 个采样点；RPU 前段候选对原 FP32 波形的 MSE 为 `2.72e−7`、
max-abs 为 `0.0122534`、相对 L2 为 `0.3056%`。CPU FP16 折叠候选本身对原 FP32
波形的相对 L2 为 `0.3684%`。此次明确改变前段精度，不声称波形逐位等价。

### 探测发现并修正的共享 RoPE 参数错误

首次候选对匹配参考的前段相对 L2 达到 `9.39%`。源码定位到
`src/fused/rpu_qwen3_model.h` 的两条无 Q/K norm 路径：SEQUENTIAL 和 KV_FIRST
把 K 的每核两个 head 合并成一个 128 维 head，实际 checkpoint 的 head_dim 为 64。
`src/ops/rpu_rope.cpp` 的 SPM launcher 明确把 head_dim 和 local_heads 分别传入，
cos/sin 表列数也按 head_dim/2 解释，因此这属于语义错误，不能归因于 FP16 舍入。

修正仅保留每个 head 的边界：K 的调用分别使用 `bs*(nkv/tp), hd` 和 `nkv/tp, hd`。
8 项 host 回归覆盖两个调度分支、每核 1/2 个 KV head 和非零位置，旧参数均能触发失败。
当前 causal codec 走 SEQUENTIAL；KV_FIRST 的共享修复另做了 31 帧双向注意力分支验证，
通过完整 planner descriptor 后，RPU 对匹配参考的相对 L2 为 `0.1738%`、max-abs
为 `0.00024414`，7 次输出逐位一致，Graph 1 项/6 次 replay 且 invariant 通过。
不把双向执行列入 codec 模型契约。

### 集成边界

独立 helper 为 `python/rpu_backend/adapters/qwen3_tts/codec_transformer.py`。
它先检查精确几何和原 CPU FP32 权重，再检查 native
`causal_decoder_has_per_head_k_rope()` 能力；旧 native 缺少该能力时返回 CPU fallback，
不创建 decoder，不吞掉其他运行错误。能力查询只标识上述原生语义修复，不是模型认证。
新 profile 独立声明 64/64 的物理 envelope，逻辑输入限定 1–63 帧；仍由现成 planner
产生完整 descriptor，未复用其他模型的已认证 envelope，也不绕过 CAPABILITY 拒绝。

helper 将资源交给现有 `_InstalledNativeResource` 和父模型 close 路径，按逻辑长度
保留一个 Graph，切换长度时淘汰旧签名，成功 capture 后才记录新签名。
原 FP32 权重保持不变，仅创建折叠后的 Half 副本；CPU FP32 卷积优化继续保留。

完整 API 首次集成另发现冷态资源峰值：Talker prefill/decode 共 2 项、Predictor 1 项、
codec 1 项，四个保留 Graph 占满默认 4 槽池，codec 的即时输出投影还需要一个临时槽。
独立 codec 测试只有一个 Graph，因此不能覆盖这个共存峰值。API 改用现成
`GraphRuntimePolicy.from_environment(graph_arena_count=5).prepare_arenas()`，在首次 RPU
权重上传/Queue 前为 TTS owner 明确准备 **4 个保留槽 + 1 个串行即时槽**，所有组件
共享同一 policy；没有更改全局默认、SDK 每槽容量或动态扩容。

该计划是进程冷态契约：若先运行其他模型并冻结了不同槽数，再在同进程切换到 TTS，
不允许在线修改池预算，需使用新进程。TTS 自身关闭重载继续使用相同的五槽计划。

五槽计划下完整 API 的 2 次预热、5 次稳态和 10 句生成全部完成。稳态中位数
**2.71871 s**、平均数 **2.72406 s**，范围 2.683–2.760 s；固定音频长 2.48 秒，
中位 RTF 为 **1.096**，尚未达到实时。作为历史对照，上一正式版本为 2.79677 s，
约下降 2.79%；这是不同运行批次的对照，不是本轮同进程配对测量。

所有完整请求的统计均明确 `codec_frontend.device=rpu`、`fallback_reason=null`；
Talker Graph 2 项、Predictor 1 项、codec 1 项，invariant 全部通过，固定输入的
codec replay 持续增长。10 句生成中中文 1 句、英文 2 句达到原 64 步上限；
10 句的生成步数和采样点数均与上一正式版本一致，未通过缩短输出或修改上限获得速度收益。完整 codec 的局部加速不能等比例代表整请求加速。

实际 helper 的完整 codec 2+5 测量为 **1103.43 → 964.77 ms（−12.57%）**。
这一次是最终 helper 配方：RoPE 相位保留原 FP32 `inv_freq`，计算 cos/sin 后转 Half；
早期手工探测的 `deepcopy().half()` 还会将 `inv_freq` buffer 转 Half，两者不可混作相同配方。
最终 helper 波形对原 FP32 波形的 MSE 为 `6.84e−7`、max-abs 为 `0.0244884`、
相对 L2 为 `0.4851%`；采样点仍为 59520。

实际 helper 已验证 1/31/63 帧边界、同长换内容、31→1→63→31、持有旧输出、
关闭重载；每个签名保持 Graph 1 项并稳定 replay，返回原输入与关闭重载后的结果逐位一致。
20 项 helper host 检查和 8 项共享 RoPE host 回归通过。

与最终 helper 相同 RoPE 配方的 CPU FP16 参考对照：1 帧相对 L2 为 `0.2181%`，
31 帧为 `0.1664%`，63 帧为 `0.1705%`；同长度换内容为 `0.1767%`。
完整波形复测与性能测量时一致。该补充只校正参考语义，不把单次冷调用作为新的性能结论。
详细计时与数值数据见[证据 JSON](evidence/qwen3_tts_rpu_coverage_r1/summary.json)。

整模型 10 句任务质量、生成耗时和与其他 RPU 组件共存的最终结论由本轮总报告汇总；
上述孤立 codec 数据不能代替整模型质量结论。探索波形的单独评分曾因错误使用运行环境
的 Transformers 版本而在导入阶段失败，未进入模型评分，未将它记为识别成功或识别失败。

完整请求的独立 ASR 内容评分仍以最终集成版为准；未完成任务质量验证前，不将孤立探测扩大成一般精度支持声明。

## 本轮的最小处理边界

保留现有 Talker/Predictor/输出头 RPU 主体和已验证的 CPU codec 优化。
FP32 codec 的 fallback 有明确的算子精度与真实几何依据。
没有为本评估新增设备程序，也没有将静态可表达性当作已经通过推理验证。

同进程、同权重切换前段的最终整句比较和独立 ASR 内容评分见[总报告](qwen3_audio_rpu_coverage.zh.md)。

最终独立官方 CPU FP32 ASR 评分：10/10 句一致；中文 49 字 CER 0%，英文 35 词 WER 0%。仅表示本组短句内容保持，未评价自然度或音色相似度。
