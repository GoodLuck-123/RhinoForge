# G0.5 Legacy 配置
[优化模板](../README.md) · [使用](#使用)

| 模板 | 输入与区别 | 输入默认 |
|---|---|---|
| [fp16.toml](fp16.toml) | 原 K1/RTC/FM 登记的 `[example]` schema；客户预处理 batch、固定 step0 | FP16，三图/P245/H32 |

支持级别为 `CONTROLLED_EVALUATION`；第二步RTC和任务质量未测。常规选型使用[六份优化模板](../README.md)。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方公开发布 | 未核实这份 G0.5 K1 `vla_rtc` release 的官方公开 HF / GitHub 权重下载 | 未核实；不从 `fp16` 运行标签推断文件 dtype | 无可替代的公开下载目录 |
| 当前实际父 | 客户 / 受控交付方提供的原始 K1 checkpoint，须配同版 release 源码、processor 和输入 | 原始 dtype 由交付清单指定；运行前按模块转 FP16 / FP32 | `/path/to/vla_rtc_release/checkpoint` → `input.checkpoint`；源码 → `input.release_root` |
| 公开量化 | 本页不使用量化资产；未核实该交付的公开量化渠道 | 不适用 | 无量化输出目录 |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../../WEIGHTS.md#precision-notation)。

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| [fp16.toml](fp16.toml) | Vision / Language 主投影 W16A16 | Action 主投影 / 设备 IO 为 W16A16；CPU AdaLN/时间条件 FP32；norm/bias 等保留浮点 | 原始 K1 + 配对 release/batch；无需转换；[完整命令](#run-fp16)，[examples/runners/g05/runner.py](../../../../examples/runners/g05/runner.py) |

<a id="quantization"></a>
## 下载与运行

<a id="run-fp16"></a>
```bash
cp examples/configs/g05/legacy/fp16.toml /tmp/g05_legacy_fp16.toml
# input.checkpoint = 原始 K1 checkpoint；input.release_root = 同交付源码。
# input.batch = 同交付预处理 batch；在 [run] 添加本次 output_dir。
vi /tmp/g05_legacy_fp16.toml
python examples/run.py --config /tmp/g05_legacy_fp16.toml
```

**无需量化转换。** 本目录只有原受控 `fp16.toml`，使用配对的客户 K1 原始权重和 batch，仅支持 FP16，没有量化输出目录。

## 使用

复制模板，填写 `input.checkpoint`、`release_root`、`batch`。batch需由相同客户release预处理器生成，满足原登记范围；运行时显式传入配置。

```bash
python examples/run.py --config /path/to/my_legacy_g05.toml --check-config
python examples/run.py --config /path/to/my_legacy_g05.toml
```

模板默认 `run.runs=3`；可设 `cpu.threads=8`、`run.warmup=2`、`run.runs=5`。

`--check-config`检查配置结构；实际权重、输入和数值由运行验证。
