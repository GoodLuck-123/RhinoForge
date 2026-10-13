# Changelog

## v1.2.0rc1

- 增加 Qwen3 ASR/TTS 独立语音接口、可选音频依赖和固定源码准备流程。
- 扩展 Qwen3.5-27B、LingBot v2 replay 及多组视觉/VLA 配置，补齐权重来源、转换和输入步骤。
- 更新 Graph 生命周期、DDR 同步、缓存与输出所有权处理，以及匹配的 Launch/main/DDR-expansion 资产组合。
- 统一 `examples/run.py` 的配置、日志与运行摘要；保留语义相同的旧脚本和配置入口。
- 按量化模块、方法和分组统一配置命名，补齐参数、模型差异和权重指南，保留旧命令路径。
- 为五类旧输入配置提供显式选择目标模板的迁移工具；RhinoVLA v1 的 caller-factory 与官方入口分开。
- 修复首次运行的生成文本显示、Wall-OSS W4 转换命令，更新运行时资产依赖、许可说明和安装文档。

升级步骤、旧路径映射与输入变更见[中文升级指南](docs/upgrading_1_2.zh.md)。
RhinoForge、Runtime 和 Rhino Launch 发行版本为 `1.2.0rc1`；Launch CMake 版本为 `1.2.0`，ABI 版本保持 `1`。

### English

- Added dedicated Qwen3 ASR/TTS APIs, optional audio dependencies and pinned source preparation.
- Expanded Qwen3.5-27B, LingBot v2 replay, vision and VLA configurations, with weight sources, conversion commands and input preparation.
- Updated Graph lifecycle, DDR synchronization, cache and output ownership, and the matching Launch/main/DDR-expansion asset set.
- Unified example configuration, logging and runtime summaries; retained compatible legacy entry points and added a migration tool for five old input schemas, with explicit target templates.
- Standardized configuration names by quantized modules, method and group size; expanded parameter, model and weight guides while preserving old command paths.
- Separated RhinoVLA v1 caller-factory and official templates; corrected first-run text output and Wall-OSS W4 commands; updated runtime asset dependencies, notices and installation guides.

See the [upgrade guide](docs/upgrading_1_2.md). RhinoForge, Runtime and Rhino Launch use release version `1.2.0rc1`; Launch uses CMake version `1.2.0` and ABI version `1`.
