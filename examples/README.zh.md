# 模型示例

[English](README.md) · [配置索引](configs/CONFIGURATIONS.md) · [通用参数](configs/PARAMETERS.md) · [命名规则](configs/NAMING.md) · [转换命令](configs/WEIGHTS.md#quantization) · [权重来源](configs/WEIGHTS.md) · [支持矩阵](../docs/model_support.zh.md)

选择模型和推荐 TOML，按模型指南准备权重与输入。所有公开配置统一使用 [`run.py`](run.py)，必须传入 `--config`。

原模型脚本保留为兼容入口。旧路径和输入迁移见 [v1.2.0 升级指南](../docs/upgrading_1_2.zh.md)，迁移命令参数见 `python examples/migrate_config.py --help`。

## 选择模型

| 模型与配置指南 | 推荐配置 |
|---|---|
| [DINOv3](configs/dinov3/README.md) | [ViT-B FP16](configs/dinov3/vit_b/fp16.toml) |
| [G0.5](configs/g05/README.md) | [3cam FP16](configs/g05/3cam/fp16.toml) |
| [Gemma4](configs/gemma4/README.md) | [E4B FP16](configs/gemma4/e4b/fp16.toml) |
| [GR00T](configs/gr00t/README.md) | [224 FP16](configs/gr00t/n1_7_droid/224/fp16.toml) |
| [HyVLA](configs/hy_vla/README.md) | [3cam FP16](configs/hy_vla/3cam/fp16.toml) |
| [HY Embodied UMI](configs/hy_embodied/README.md) | [UMI FP16](configs/hy_embodied/umi/fp16.toml) |
| [Llama](configs/llama/README.md) | [3.2-1B FP16](configs/llama/3_2_1b/fp16.toml) |
| [Pi0.5](configs/pi05/README.md) | [3cam W8A16](configs/pi05/3cam/w8a16.toml) |
| [Qwen3](configs/qwen3/README.md) | [0.6B FP16](configs/qwen3/text/0_6b/fp16.toml) |
| [Qwen3.5 text](configs/qwen3_5/README.md) | [0.8B FP16](configs/qwen3_5/text/0_8b/fp16.toml) |
| [Qwen3.5 vision](configs/qwen3_5/README.md) | [2B FP16](configs/qwen3_5/vl/2b/fp16.toml) |
| [Qwen3-VL](configs/qwen3_vl/README.md) | [2B FP16](configs/qwen3_vl/vl/2b/fp16.toml) |
| [RhinoVLA](configs/rhinovla/README.md) | [v1 FP16](configs/rhinovla/v1/official_fp16.toml) |
| [Wall-OSS base](configs/wall_oss/README.md) | [3cam 224 FP16](configs/wall_oss/base/3cam/224/fp16.toml) |
| [Wall-OSS RTC](configs/wall_oss/rtc/README.md) | [FP16](configs/wall_oss/rtc/fp16.toml) |
| [Wall-OSS sorting](configs/wall_oss/sorting/README.md) | [W8A16](configs/wall_oss/sorting/w8a16.toml) |
| [InternVLA](configs/internvla/README.md) | [NavDP System2 W8](configs/internvla/navdp/w8a16_system2.toml) |
| [LingBot v2](configs/lingbot/v2/README.md) | [3cam T3 W8A16](configs/lingbot/v2/3cam_t3/w8a16.toml) |
| [LingBot v1](configs/lingbot/README.md) | [4B CPU Vision / FP16](configs/lingbot/v1/4b/fp16_cpu_vision.toml) |
| [HALO](configs/halo/README.md) | [FP16 Expert](configs/halo/action_expert/fp16.toml) |

Qwen3.5 35B MoE 仅支持已登记的受控配置。

## 运行

安装 backend 后，复制所选 TOML，按模型指南填写资产路径、输入和 `run.output_dir`。

```bash
cp examples/configs/g05/3cam/fp16.toml /tmp/g05.toml
# 按配置指南修改路径和参数
python examples/run.py --config /tmp/g05.toml --check-config
python examples/run.py --config /tmp/g05.toml
```

`--check-config` 无需 Torch/设备，只检查配置；`--help` 显示统一命令的选项。ACC、文本长度、CPU 线程和采样次数在同一 TOML 中修改，合法值见模型指南。

<a id="logging-and-output"></a>
<a id="日志与输出"></a>
## 日志与运行摘要

在 `[run]` 同级加入以下配置；省略时使用相同默认值：

```toml
[logging]
level = "ERROR"
summary = true
```

<a id="日志等级对照"></a>
`level` 接受下列大写值，按等级过滤诊断信息；`summary=true` 时各级均显示模型配置摘要。

| 等级 | 显示内容 |
|---|---|
| `CRITICAL` | 致命错误；普通 ERROR 隐藏 |
| `ERROR` | ERROR、CRITICAL；默认值 |
| `WARNING` | 增加告警及对应等级的 fallback/Graph 信息 |
| `INFO` | 增加加载过程、普通 `print()` 和生成文本 |
| `DEBUG` | 增加已有 Graph 诊断及原生 TRACE |

普通 `print()` 按 INFO 处理；可识别标签按其等级过滤，Python 异常继续传播。`summary=false` 关闭模型摘要；没有 `logging.file` 字段。原生输出存入 `run.output_dir` 的 `native-*.stdout.log` / `native-*.stderr.log`，进程结束时按等级回放。

`[MODEL]` 列出输入、设备配置、CPU 线程和输出位置。

## 查看结果

`run.output_dir` 指定输出位置；部分配置创建子目录。查看生成文件中的检查结果和产物路径。多长度结果位于各 `text_N/`。

[运行库准备](../docs/runtime_assets.md)

## Qwen3 语音

- [`qwen3_asr.py`](qwen3_asr.py)：本地单声道 16 kHz PCM16 WAV 转文本。
- [`qwen3_tts.py`](qwen3_tts.py)：中/英文短文本与单声道 24 kHz 参考音频生成 WAV。

安装 `audio` 可选依赖，按[语音指南](../docs/qwen3_audio.zh.md)准备固定权重、源码和
输入，再执行对应命令。这两个脚本使用独立 API，不经过 TOML runner。
