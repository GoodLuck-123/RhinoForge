# RhinoVLA v1 官方 FP16
[官方模板](official_fp16.toml) · [旧 caller-factory 配置](fp16.toml)

官方非 RTC 路径，固定 a3 源码和 checkpoint；2图、P171、H30、十步去噪，动作完整72D/有效64D。与 [v3 factory](../v3/README.md)使用不同权重和输入。

<a id="weights"></a>
<a id="weight-sources"></a>
## 官方来源与原始存储精度

| 资产 | 官方下载 / 源码 | 原始存储精度与版本 |
|---|---|---|
| `rhinovla_pretrain.ckpt` | [HuixiAI/RhinoVLA](https://huggingface.co/HuixiAI/RhinoVLA) | 固定 revision `7f3708650e302f395e7c63607b48b56fb96b8076`；HF 元数据未列 `.ckpt` 各张量 dtype，**原始存储类型待确认**。本页 FP16 是运行精度 |
| 配套源码 | [HUIXI-AI/RhinoVLA](https://github.com/HUIXI-AI/RhinoVLA) | 必须匹配 `a3edf0c1addf4bf849192642bff3ad15559db3f4`；数据 parquet/fixture 也须按准备脚本配对 |

官方下载与源码准备如下；已有相同版本时不需重新下载。这里不生成量化权重：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
RHINO_ROOT="$RPU_MODEL_CACHE/rhinovla-controlled"
git clone https://github.com/HUIXI-AI/RhinoVLA.git "$RHINO_ROOT/source"
git -C "$RHINO_ROOT/source" checkout a3edf0c1addf4bf849192642bff3ad15559db3f4
hf download HuixiAI/RhinoVLA rhinovla_pretrain.ckpt --revision 7f3708650e302f395e7c63607b48b56fb96b8076 --local-dir "$RHINO_ROOT/weights"
```

<a id="weight-recipes"></a>
## 模块精度与配方

| 配置 | 模块实际 W/A 精度 | 输入父权重 → 转换方式与命令 |
|---|---|---|
| `official_fp16.toml` | 官方非 RTC Vision/Text/Action 的设备投影为 W16/A16；CPU fixture/参考生成独立采用 FP32 | 上述固定 v1 checkpoint → 官方加载后转 FP16；无低比特量化或量化导出 CLI。先准备同一源码的 fixture，再按[使用](#使用)运行 |

输入 fixture 由同一固定源码和 checkpoint 准备；v3 W8 使用另一套 96D 模型。

<a id="quantization"></a>
## 下载与运行

**无需量化转换。** 官方配置使用 `official_fp16.toml`；配对 fixture 使用同一源码和 checkpoint。v3 W8 属于另一客户模型，不能替代本 v1/72D checkpoint。

## 旧 caller-factory 入口

`fp16.toml` 保留原有 caller-factory 含义：调用者提供 checkpoint、配对请求、
可信 `input.runtime_factory` 和 `input.factory_config`。已有集成继续使用这些资产，
切换官方路径时另行准备本页的固定源码和 fixture。迁移步骤见
[升级指南](../../../../docs/upgrading_1_2.zh.md)。

## 使用

将已准备的官方完整资产登记为 `rhinovla-controlled`。目录需要包含 `source/`、`weights/rhinovla_pretrain.ckpt`、`fixture/cpu_fixture.pt`，只放 checkpoint 不足以运行。模板内 `trust_model_code=true`、`num_steps=10`，输入与噪声从同一固定 fixture 读取。

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
python examples/run.py --config examples/configs/rhinovla/v1/official_fp16.toml --check-config
python examples/run.py --config examples/configs/rhinovla/v1/official_fp16.toml
```

`$RPU_MODEL_CACHE/rhinovla-controlled`指向上述完整资产。可修改 `run.warmup/runs/output_dir` 和 `cpu.threads`；可设 `cpu.threads=8`、`run.warmup=2`、`run.runs=5`，模板默认3次正式运行。非 RTC、固定官方 source/checkpoint 与十步合同保持不变。
