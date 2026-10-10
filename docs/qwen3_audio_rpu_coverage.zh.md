# ASR / TTS：扩大 RPU 驻留范围

后续将 ASR 末尾 GELU/proj2 纳入现有 GraphCache 的结果见[音频图优化报告](qwen3_asr_graph_tail.zh.md)。本页保留本轮原始数据。

2026-10-10，分支 `feat/qwen3-audio-rpu`，起点 `f8f0d9f`。沿用冻结的
Qwen3-ASR-0.6B 和 Qwen3-TTS-12Hz-0.6B-Base 检查点、已审核上游源码及 R1
RhinoForge-runtime-v1.1.0 资产包。未新增设备程序或依赖；新增 host 包装复用已有算子。
ASR 4 CPU 线程、TTS 8 CPU 线程、8 RPU cores；编译和实测持共享设备锁串行执行。

## 当前各阶段运行位置

| 模型 | 阶段 | 最终设备与说明 |
|---|---|---|
| ASR | 读音频、mel 特征、分块 | CPU |
| ASR | 三层音频卷积及层间 GELU、末尾布局 | CPU；FP16 舍入权重，FP32 卷积后转 FP16 |
| ASR | conv_out、位置相加、有效行选择 | RPU；有效行通过 view/slice 选择，没有 CPU 拼接 |
| ASR | 18 层音频 encoder：attention、norm、GELU、Linear | RPU 连续 Graph；包括最后 ln_post、proj1，合计 109 个逻辑 Linear |
| ASR | 最后 ERF GELU、proj2 | RPU；与 conv_out 合计全部 111 个音频 Linear 都在 RPU |
| ASR | prefill 文本 embedding 与音频特征组装 | RPU；复用已有 gather/assembly 实现 |
| ASR | 28 层文本 decoder、词表输出头 | RPU；decode 输出头在 Graph 内，prefill 仅计算最后位置 |
| ASR | 单 token embedding、finite 检查、argmax、文本解码 | CPU；查表后上传，保留可见的数值校验 |
| TTS | Talker 28 层、Predictor 5 层 | RPU；attention/norm/SiLU/Linear 在融合路径中 |
| TTS | 16 个输出头 | RPU；直接消费当前 RPU hidden |
| TTS | codec 前段输入投影、8 层 Transformer、输出投影 | **新增 RPU FP16 路径**，Linear 使用 FP32 累加 |
| TTS | codec 码本查表与求和、卷积/上采样、SnakeBeta、波形输出 | CPU FP32 |
| TTS | 文本/音频码 embedding、生成控制和采样选择 | CPU FP32 |
| TTS | 参考音频处理、speaker encoder | CPU；同一参考音频复用 speaker embedding |

“驻留”指上述连续段计算不再为了下一个算子把结果送回 CPU，不表示整个请求没有
传输。ASR 最终音频特征仍读回一份用于 finite 检查，但后续组装消费保留在 RPU 的特征。
TTS 的 codec 前段从 CPU 码本输出接收数据，并将结果交给 CPU 波形解码后段。

## 为什么仍保留 CPU

- **ASR 卷积**：现有 RPU Conv2d 的对齐和 padding 成本较大，既有实机路径为
  392.69 ms，对照 CPU 94.94 ms；本轮未重复该无收益独立路径。当前 CPU 卷积已采用
  channels-last。后续若迁移，应评估连续卷积/激活融合，而非逐层下发。
- **ASR 单 token 查表**：同进程消融 2+5，CPU 查表方案 255.76 ms，RPU 查表
  257.88 ms（均为完整请求中位数）；保留 CPU 小查表，prefill 组装仍在 RPU。
- **TTS 后半段 codec**：现有 FP32 Linear、matmul、卷积/转置卷积、norm、Snake
  运算没有保持原精度的完整原生 RPU 路径。部分实际卷积几何还不满足通道对齐或 SPM
  容量限制。单纯 `.to('rpu')` 会触发内部 CPU fallback，增加传输。
- **TTS embedding 和 speaker**：保留现有 FP32 运算配方；先转换各表再相加会改变
  舍入语义。需要另行做精度与任务质量评估，不能当成等价设备搬迁。

详细算子和几何证据见 [TTS 覆盖评估](qwen3_tts_rpu_coverage.zh.md)。当前不是任意
算子失败都自动重试的通用 fallback：已知不适合的阶段明确选 CPU，未知执行错误仍报告。

## 本次实现的关键边界

ASR 新增 `gelu_erf` 的 eager host 包装，输入限定为 contiguous RPU FP16、元素数为
8 的正整数倍。它使用已有 ERF 算子及有界 SPM 临时区，返回独立存储，不接受 Graph
capture；encoder 内仍用原融合 SPM 路径。预先驻留的位置表不超过 8 个 100-frame
chunk；只有末尾 chunk 有 padding，全部有效行形成连续前缀。测试覆盖短输入、整块、
跨块和 8 秒上界。旧 native 缺少必需接口时 ASR 在加载权重前拒绝。

TTS 新前段不是 FP32 等价迁移：LayerScale 在 FP32 中折入投影输出行，再转换 FP16；
RoPE 保留原 FP32 inv_freq 计算相位。逻辑长度限定 1–63 帧，严格小于原 72-frame
滑窗。原 FP32 权重保留。修复共享 decoder 无 Q/K norm 分支把多个 K head 错并为
一个宽 head 的 RoPE 参数；SEQUENTIAL 和 KV_FIRST 均有测试和实机参考对照。

TTS 使用已有 owner runtime policy，在首次设备分配前申请 **5 个 Graph arena**：
Talker prefill/decode 两个、Predictor 一个、codec 一个，再加串行即时算子一个。
初次联合测试揭示默认 4 槽不足；已用有限预算修复，未扩大框架全局默认或绕过容量检查。
不同长度只淘汰对应旧签名，稳定长度保持 BUILD→REPLAY。已冻结为不兼容预算的进程
必须重启。旧 native 没有逐 head K RoPE 能力时，codec 前段明确保留 CPU，
`execution_stats()` 返回设备和 fallback 原因，不吞运行异常。

## 完整请求与质量

固定输入采用 2 次预热、5 次测量。ASR 为同一 4.20394 秒录音、11 个 token；
TTS 为同一句中文、32 步生成、2.48 秒音频。加载不计时，结果检查在计时外。

| 同进程对照 | 原 CPU 边界 | 最终扩大 RPU 驻留 | 变化 |
|---|---:|---:|---:|
| ASR 完整请求，中位数 | 255.38 ms | 261.76 ms | +2.50%，没有提速 |
| TTS 完整请求，中位数 | 2.80991 s | 2.69821 s | **−3.98%** |

ASR 均值为 255.27→259.87 ms；两轮和后续带计时 hooks 的结果存在几毫秒波动。
TTS 同一份驻留权重只切换 codec 前段，每个模式各做 2+5；均值 2.81183→2.70052 s，
固定 32 步、59520 个采样点，最终中位 RTF **1.088**，尚未实时。
独立标准 benchmark 得到 TTS 中位 2.71871 s（aggregate RTF 1.098），与配对结果分别记录。

### 分阶段耗时

以下为带阶段计时的 5 次测量均值，包含传输和下发；不能视作设备纯计算时间。
ASR 单独分解，TTS 在上述同进程切换中分解，不将不同轮次阶段相加构造完整请求中位数。

| ASR 阶段 | 本轮，ms |
|---|---:|
| 特征提取/分词，CPU | 3.43 |
| 三层卷积，CPU | 75.12 |
| 音频 encoder + ln_post/proj1，RPU | 6.17 |
| conv_out / proj2，RPU | 1.20 |
| 音频剩余激活/位置/布局，CPU + RPU | 17.90 |
| 文本 prefill，RPU | 13.30 |
| prefill 输出头，RPU | 2.44 |
| 文本 decode + fused 输出头，RPU | 113.46 |
| 其余组装/查表/校验/调度 | 21.40 |
| 诊断总计 | 254.42 |

| TTS 阶段 | CPU 前段基线，ms | RPU 前段，ms |
|---|---:|---:|
| Talker，RPU | 298.22 | 298.22 |
| Predictor，RPU | 1061.81 | 1060.45 |
| 16 输出头，RPU | 207.24 | 206.74 |
| **codec 前段，CPU→RPU** | **120.79** | **4.01** |
| codec 后半段，CPU | 917.61 | 924.34 |
| 其余参考读取/生成控制/调度 | 206.15 | 206.76 |
| 诊断总计 | 2811.83 | 2700.52 |

codec 前段减少约 **96.7%**，完整 codec 均值 1038.40→928.34 ms，减少 **10.6%**。
主要剩余耗时为 Predictor 和 CPU codec 后半段；本轮不改变生成步数来缩短时间。

### 内容与数值

ASR 最终 20/20 录音的 token 与原正式版本一致：中文 CER **1.818%**（165 字），
英文 WER **5.645%**（124 词）。0.5 秒和 8 秒边界通过。

独立官方 CPU FP32 ASR 对 TTS 音频转写：**10/10 句内容一致**。中文 CER 0%
（5 句、49 字），英文 WER 0%（5 句、35 词），与上一正式版本相同。这是小样本
内容一致性，不是音质评分；测试前将 24 kHz 波形重采样为 16 kHz。

TTS 10 句生成步数、音频长度均与旧版一致，其中 3 句达到 64 步上限。
固定句波形与原 FP32 codec 前段相比，MSE `6.841e−7`、max-abs `0.0244884`、
相对 L2 `0.4851%`，输出有限且长度不变。新前段改变了精度配方，不声称波形逐位等价。


ASR 首轮“全部 embedding 上 RPU”测得 259.37→267.43 ms；单 token 查表消融后
回到 CPU，但最终同权重对照仍有小幅额外开销。不能把驻留范围扩大称作已提速。
音频连续段最终特征与旧路径形状相同、全部有限；MSE `1.079e−10`、max-abs
`0.00012207`，相对 L2 `0.05491%`。这些仅为数值观察，不替代识别质量门槛。

本轮没有重测 Thor。历史 TensorRT 主体路径为 ASR 154.21 ms、TTS 1.057 s；
TTS 两机精度、采样和步数不匹配，不给出等工作量硬件倍数。见
[Thor TensorRT 报告](qwen3_audio_thor_tensorrt.zh.md)。

## 验证与复现

- ASR 49 项 host 用例按受影响范围分批通过；最终 embedding 修改另有定向回归。
- TTS 原 33 项 host 用例在完整 R1 Python 环境通过；冷态 arena 规划改动做了定向回归。
- 新 TTS helper 20 项、共享 RoPE 8 项 host 用例通过；ERF GELU 2 项硬件用例通过。
- ASR 最终 20 条质量及 0.5/8 秒边界通过，音频 Graph 1 项、文本最多 2 项，invariant 正常。
- TTS 独立 helper 的 1/31/63 帧、同长换内容、长度往返、输出持有及关闭重载通过。
- TTS 完整 API 2+5+10 的四张 Graph 共存、不同长度、稳定 replay、独立波形通过。

结构化汇总见[本轮证据 JSON](evidence/qwen3_audio_rpu_coverage_r1/summary.json)。
标准完整请求测试入口为 `examples/benchmark_qwen3_audio.py`；R1 本轮目录为
`/home/agi/rhinoforge_audio_runtime/rpu_coverage_20261010`。原始固定 manifest 位于
`benchmark_20261009`，独立 TTS 内容评分使用其中冻结官方 ASR CPU FP32 参考及隔离
Transformers 4.57.6 环境。没有用本轮改动的 ASR 自行给 TTS 评分。

已更新 R1 正式目录 `/home/agi/rhinoforge_audio_runtime` 的对应 Python 文件、benchmark
和 native 扩展，部署前核对候选与工作树文件一致，替换时备份旧版并校验文件哈希。
旧版备份位于 `backup_f8f0d9f_audio_coverage_20261010`；新 native SHA256 为
`b528fd132fb4f26bfc2301137f08d37599f8117fa5b5a60aaa37099d085b8a40`。
部署后 ASR 与 TTS 均通过换内容/长度往返、输出重复性和同进程关闭重载；
TTS 的三个组件均保持各自有界缓存及 invariant。部署与生命周期结果记录在证据 JSON。

限制：短语料的 ASR/CER/WER 和 TTS 内容转写不代表通用准确率、自然度或音色相似度。
未做人工音质评测、长文本或流式认证，TTS 仍受 64 步上限限制，部分句子会达到上限。
