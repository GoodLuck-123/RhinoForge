# 模型运行与调试

**简体中文** | [English](model_testing.md)

从[模型示例](../examples/README.zh.md)选择 TOML。运行器位于 `examples/runners/`，共享配置与执行代码位于 `examples/common/`。

## 运行

```sh
python examples/run.py --config examples/configs/qwen3/text/0_6b/fp16.toml --check-config
python examples/run.py --config examples/configs/qwen3_vl/vl/4b/w8a16_text_head_vision.toml
python examples/run.py --config examples/configs/pi05/2cam/w8a16_action_nvfp4.toml
```

按模型指南填写 checkpoint 和输入，替换占位路径。`input.prompt_file` 和 `input.prepared_input_safetensors` 的相对路径按配置文件所在目录解析；其他资产路径按模型指南填写。`--check-config` 检查配置，不加载权重或访问设备。量化权重须携带加载器要求的元数据，见[量化](quantization.zh.md)。

预处理输入使用对应 checkpoint 的 processor 产生的 CPU 张量字典，形状与组件须符合所选模型要求。

## 执行参数

- `run.warmup` 默认两次调用，让新 Graph 完成 BUILD 和 REPLAY；`run.runs` 设置后续调用次数。
- `warmup >= 2` 时，省略 `warmup_decode_steps` 或令其等于 `input.decode_steps`。显式 `warmup=0/1` 可做缩短 decode 的冷路径诊断。
- decode 步数不含 prefill 产生的首 token。EOS 可提前结束；固定长度运行时，两阶段均设 `stop_on_eos=false`。
- `run.output_dir` 指定输出目录；在生成文件中查看输出位置和 Graph 状态。

## 诊断调用

`--profile` 或 `[run].profile=true` 在常规运行后增加一次 Torch profiler 调用。按需启用 shape、memory 或 stack。跟踪可能包含应用数据，保存在源码仓库外。

公开入口拒绝 `hwperf=true`。设备问题可使用所安装运行时的 profiler，并记录匹配的资产和输入 shape。参数见[运行时配置](runtime_config.zh.md)，检查范围见[模型验证](validation_policy.zh.md)。

## Qwen3 语音

安装 `audio` 可选依赖，使用 `examples/qwen3_asr.py` 和 `examples/qwen3_tts.py`。
下载权重、准备 TTS 源码及完整命令见[语音指南](qwen3_audio.zh.md)。
这些独立脚本不经过文本/视觉 TOML runner。
