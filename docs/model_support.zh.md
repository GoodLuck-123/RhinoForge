# 模型入口与运行范围

**简体中文** | [English](model_support.md)

从[模型示例](../examples/README.zh.md)选择 TOML，按模型指南准备权重和输入，执行 `python examples/run.py --config PATH --check-config` 后运行。检查命令只验证配置。

支持范围按 checkpoint、精度、核数、输入形状和长度确定，由加载器与运行时准入检查执行。受控入口还需模板声明的 opt-in；详细参数见[配置索引](../examples/configs/CONFIGURATIONS.md)。

权重与请求数据由调用方提供。安装依赖见[模型资产](model_assets.zh.md)和[运行时资产](runtime_assets.zh.md)；调用方式见[运行与调试](model_testing.zh.md)、[API](api_reference.zh.md)和[量化](quantization.zh.md)。

## Qwen3 语音

`Qwen3ASRModel` 和 `Qwen3TTSModel` 提供独立的 ASR/TTS 入口，由 CPU 和设备组件
共同执行。固定资产、输入范围、源码准备和进程所有权见[语音指南](qwen3_audio.zh.md)。
这两个接口不经过文本/视觉 TOML runner 或 CausalLM/VLM registry。
