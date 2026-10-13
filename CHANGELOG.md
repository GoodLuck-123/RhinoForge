# Changelog

## v1.3.0.dev0

开发线已进入 1.3；当前仅提供源码。

Development has moved to 1.3; this line currently provides source only.

## v1.2.0rc1

- 增加 Qwen3.5-2B 的 MLP W4、attention/GDN W8 混合量化转换和三份示例配置；转换保留输入浮点权重的原始精度。
- 补齐示例参数、权重与输入步骤、旧路径映射，以及配置导出和执行工具。
- 恢复规划摘要协议、显式原生扩展加载、Graph 结构导出和缺失的诊断类别。
- 补充模型入口、计算核拓扑、缓存和 greedy token API 文档。
- 增加 Qwen3 ASR/TTS 独立语音接口、可选音频依赖和固定源码准备流程。
- 扩展 Qwen3.5-27B、LingBot v2 replay 及多组视觉/VLA 配置，补齐权重来源、转换和输入步骤。
- 补齐 RhinoVLA v1/v3、InternVLA 和 HyVLA 输入准备命令、V3 内置工厂及九组独立配置。
- 恢复共享 mask、one-shot 队列复用、缺失的线性算子选型和逻辑 Tensor 调试接口；Graph snapshot 校验表绑定当前算子包。
- 更新 Graph 生命周期、DDR 同步、缓存与输出所有权处理，以及匹配的 Launch/main/DDR-expansion 资产组合。
- 统一 `python -m examples.run --config <TOML>` 的配置、日志与运行摘要。
- 按量化模块、方法和分组统一配置命名，补齐参数、模型差异和权重指南。
- 修复首次运行的生成文本显示、Wall-OSS W4 转换命令，更新运行时资产依赖、许可说明和安装文档。

安装步骤与输入要求见[中文升级指南](docs/upgrading_1_2.zh.md)。
RhinoForge、Runtime 和 Rhino Launch 发行版本为 `1.2.0rc1`；Launch CMake 版本为 `1.2.0`，ABI 版本保持 `1`。

### English

- Added Qwen3.5-2B mixed MLP W4 / attention-GDN W8 conversion and three example configurations; conversion preserves source floating-point precision.
- Completed example parameters, weight/input preparation, path aliases and configuration export/run tools.
- Restored planner digest protocols, explicit native-extension loading, structural Graph export and diagnostic categories.
- Expanded model entry, core topology, cache and greedy token API documentation.
- Added dedicated Qwen3 ASR/TTS APIs, optional audio dependencies and pinned source preparation.
- Expanded Qwen3.5-27B, LingBot v2 replay, vision and VLA configurations, with weight sources, conversion commands and input preparation.
- Added RhinoVLA v1/v3, InternVLA and HyVLA input preparation commands, a built-in V3 factory and nine standalone configurations.
- Restored shared masks, one-shot queue reuse, missing linear operator selections and logical tensor debugging; bound Graph snapshot contracts to the current operator assets.
- Updated Graph lifecycle, DDR synchronization, cache and output ownership, and the matching Launch/main/DDR-expansion asset set.
- Unified example configuration, logging and runtime summaries through `python -m examples.run --config <TOML>`.
- Standardized configuration names by quantized modules, method and group size; expanded parameter, model and weight guides with current template paths.
- Corrected first-run text output and Wall-OSS W4 commands; updated runtime asset dependencies, notices and installation guides.

See the [upgrade guide](docs/upgrading_1_2.md). RhinoForge, Runtime and Rhino Launch use release version `1.2.0rc1`; Launch uses CMake version `1.2.0` and ABI version `1`.
