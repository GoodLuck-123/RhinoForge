# ASR：将末尾 GELU / proj2 纳入 GraphCache

2026-10-10，基线 `a8da29e`，分支 `feat/qwen3-audio-rpu`。保持已冻结的
Qwen3-ASR-0.6B 检查点、FP16 边界和 FP32 Linear 累加、0.5–8 秒输入、单请求、
最多 64 个生成 token，以及原 R1 runtime-v1.1.0 / Launch / 算子资产。
本轮只修改 ASR 音频图尾部，不改变 TTS 路径、CPU 卷积或文本解码。

## GraphCache 覆盖变化

此前已有三张常驻图：音频 encoder 一张，文本 prefill / decode 两张。
音频 encoder 内含 18 层、最后 LayerNorm、proj1；其后的 ERF GELU 和 proj2
虽然在 RPU 上执行，仍是独立即时调用。

现在把 **ERF GELU 和 proj2 接入原音频图**，不增加 GraphCache 实例或 arena 数量。
音频图内包含 110 个逻辑 Linear，conv_out 为图外第 111 个。公开调用方式保持不变。
同长度时只首次 BUILD，后续 REPLAY；长度变化沿用原单条目淘汰规则。

不能直接用 `with graph.capture(...)` 包住旧的两个调用：旧 `gelu_erf` 使用即时 DMA，
旧 `linear_with_core_count` 明确拒绝捕获具有新输入/输出地址的调用。本轮保持这些
防护，复用融合 encoder 的原生 host 调度路径。

- 新内部 setter `siglip_set_audio_projection` 仅接收精确 ASR 模式、RPU FP16
  `[1024,896]` 已转换权重及 `[1024]` bias，只允许安装一次。
- 新投影纳入物理执行计划的 Linear route，尾部开关参与模型布局 hash。
- proj1 的中间 DDR 缓冲由模型按 chunk 长度持有；GELU 复用最后一层已不再使用的
  norm SPM 区，随后 proj2 写入完整输出。保留 FP16 舍入边界和同一 ERF 公式。
- 最终输出每次分配独立存储，并在每次 forward 更新 mutable DMA 基地址，避免
  快速重放写回旧输出。内部固定地址缓冲随 handle 保留，清图后再销毁 handle。
- 输入组装、conv_out 和位置相加仍在图外，CPU finite 检查也保留。GraphCache
  降低重复下发开销，仍有实际数据搬运；本次没有声称整个请求合成一张图。

旧 native 没有新增 setter 时，在加载完整权重前拒绝，需配套重编译。

## 速度

R1 8 RPU cores，CPU 4 线程，固定 4.20394 秒录音、11 个生成 token。
持共享锁串行执行，排除加载，每条路径 2 次预热、5 次测量。

首次同进程顺序对照，每条路径重新加载干净模型，附轻量级阶段计时：

|阶段|原即时尾部|Graph 尾部|
|---|---:|---:|
|音频 encoder + GELU + proj2，均值|7.054 ms|6.349 ms|
|完整音频阶段，均值|101.091 ms|106.412 ms|
|完整请求，中位数|253.341 ms|260.703 ms|

目标段减少 **0.705 ms（10.0%）**，但整句变慢。按该轮分解，音频图外部分
增加约 6.026 ms；不能把这部分自动归因于图尾部，也不能将图内收益冒充整句收益。
因此增加一次针对该波动的同模型交替对照。

补充对照保持同一个公开 ASR 模型、同一份 CPU 卷积和文本权重，只切换由相同原始权重
分别安装的两个音频 handle。每条路径 2 次预热、5 次交替测量。为维持默认四槽预算，
切换前清除离开的音频图，在计时外准备进入路径的音频图；计时内仍完整执行公开
`transcribe()`，包含卷积、音频图重放、文本图重放与结果返回。额外的诊断准备调用
明确不计入完整 API 时间，公开生产路径没有这些切换操作。

|阶段|原即时尾部|Graph 尾部|
|---|---:|---:|
|音频 encoder + GELU + proj2，均值|7.204 ms|6.383 ms|
|CPU 三层卷积，均值|76.290 ms|75.175 ms|
|完整请求，中位数|255.245 ms|252.968 ms|
|完整请求，均值|256.791 ms|255.251 ms|

局部节省 **0.822 ms（11.4%）**，与顺序对照一致。完整请求本轮中位数下降约
**0.9%**，但 5 组相邻样本差值为 −0.55、−12.00、+6.08、+1.94、−3.17 ms，
CPU 卷积仍有波动。因此结论是**GraphCache 明确降低目标段耗时，整句收益较小，
尚不足以声称稳定的大幅提速**。不采用上轮 261.76 ms 与本轮更低数值跨轮相减。


## 正确性和缓存

- 49 项原 ASR host 测试通过；新增图尾输出宽度、同长输入更新和 proj1/proj2
  计数测试单独通过。
- 固定输入最终音频特征 `[55,1024]` 与原路径 **MSE / max-abs / relative L2 均为 0**，
  形状、精度和 finite 正常；该单样本结果不扩展为任意输入保证。
- 反转原音频形成同长度不同内容：输出更新，持有旧输出不变，再回到原输入结果复现。
- 两条路径各 7 次完整请求，音频缓存一直为 1，replay 从 0 增长到 6；文本图最多 2，
  两类缓存 invariant 均为真。
- 20 条语料输出 token 均与上一版一致：中文 CER 1.818%（165 字），英文 WER
  5.645%（124 词）。0.5 秒和 8 秒边界、不同长度后回到原输入通过。

原始结果在 R1 `/home/agi/rhinoforge_audio_runtime/asr_graph_20261010`，
`paired.json` 为顺序对照，`interleaved.json` 为消除 CPU 卷积实例差异的补充对照。
[结构化摘要](evidence/qwen3_asr_graph_tail_r1/summary.json)记录原始测量序列与质量结果。

已更新 R1 正式运行目录的 ASR 两个 Python 文件和 native 扩展。旧版本备份为
`/home/agi/rhinoforge_audio_runtime/backup_a8da29e_asr_graph_20261010`。
新 native SHA256 为 `3965cbfb3621f83a415ebd67aec42be63cb2ccd87b38025982ceb706e143c9e0`。
正式目录验证了换内容、长度往返及关闭重载，结果复现、Graph invariant 正常。
部署前后的文件身份和生命周期结果记录在证据摘要中。

未迁移 CPU 卷积或输入组装，也未重测 Thor/TTS。上述短输入测试不扩大原模型支持范围。
