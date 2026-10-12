# Qwen3 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；推荐 [0.6B FP16](text/0_6b/fp16.toml)。本目录是纯文本 Qwen3；图文模型见 [Qwen3-VL](../qwen3_vl/README.md)，混合架构见 [Qwen3.5](../qwen3_5/README.md)。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源与存储类型

下表的存储精度来自发布仓库的 `config.json` 和 HF safetensors 元数据（2026-10-10 核对），**不是 TOML 的运行精度**。目录均相对于 `$RPU_MODEL_CACHE`；下载完整仓库，包含 config、tokenizer 和全部权重分片。

| 权重 / 用途 | 磁盘存储精度 | 发布者与下载来源 | 下载到的目录 |
|---|---|---|---|
| Qwen3-0.6B：普通 FP16/W4 父、W8 转换输入 | BF16 | Qwen 官方 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-0.6B) | `Qwen3-0.6B` |
| Qwen3-1.7B：普通 FP16/W4 父、W8 转换输入 | BF16 | Qwen 官方 [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-1.7B) | `Qwen3-1.7B` |
| Qwen3-4B：普通 FP16/W4 父、W8 转换输入 | BF16 | Qwen 官方 [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-4B) | `Qwen3-4B` |
| Qwen3-8B：普通 FP16/W4 父、W8 转换输入 | BF16 | Qwen 官方 [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-8B) | `Qwen3-8B` |
| Qwen3-14B：W4 父、W8 转换输入（无 FP16 运行） | BF16 | Qwen 官方 [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-14B) | `Qwen3-14B` |
| Qwen3-32B：W4 父、W8 转换输入（无 FP16 运行） | BF16 | Qwen 官方 [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-32B) | `Qwen3-32B` |
| 4B Instruct：独立 FP16 模板 | BF16 | Qwen 官方 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `qwen3-4b-instruct` |
| 0.6B Instruct：历史已供模板 | 原包存储 dtype 待确认；下列公开包为 BF16 | 历史下载定义指向第三方 [rd211/Qwen3-0.6B-Instruct](https://huggingface.co/rd211/Qwen3-0.6B-Instruct)，未证明与已供 `instruct-local` 同值 | `qwen3-0.6b-instruct`，复现须原包 |
| 本仓库 W8 产物 | Text7/head 为 INT8 + FP16 scale；其余 FP16 | 本仓库从相应浮点父转换；没有可替代此格式的官方量化下载地址 | 0.6–14B：`qwen3_lmhead_int8_ckpts/qwen3-{SIZE}b-w8a16-lmhead-int8`；32B：`Qwen3-32B-w8a16` |

14B/32B 的官方 BF16 父用于量化，并不开放这两个尺寸的 FP16 运行。既有 14B W8 包与新转换目录分别记录来源。

<a id="weight-recipes"></a>
## 模块精度与配方

**Text7** 指每层 Attention 的 `q_proj/k_proj/v_proj/o_proj` 与 MLP 的 `gate_proj/up_proj/down_proj`。W8 为逐输出通道对称 INT8；W4 为对称 INT4、沿输入维度每32个值一组（G32）。`A16` 指 FP16 激活，和可独立选择的累加精度不同。

| config 配方（同一行包含各尺寸） | 运行时模块权重 / 激活 | 输入资产 | 获取方式、源码与完整命令 |
|---|---|---|---|
| `text/{0_6b,1_7b,4b,8b}/fp16.toml` | Text7、embedding、norm、head 为 FP16；A16 | 对应官方 BF16 父 | 不量化；加载时转 FP16；[examples/runners/qwen3/runner.py](../../runners/qwen3/runner.py)；[下载并运行](#cmd-fp16) |
| `text/{0_6b,4b}/fp16_instruct.toml` | 同上；使用独立 Instruct 权重 | 4B 官方 Instruct；0.6B 原已供包 | 不量化；[4B 下载 / 0.6B 原包运行](#cmd-instruct) |
| `text/*/w8a16.toml`，0.6/1.7/4/8/14/32B | Text7 **和 head W8**；embedding/norm FP16；A16 | 相应官方 BF16 父（历史14B限制见上） | 离线；[python/rpu_backend/quant/convert_qwen3.py](../../../python/rpu_backend/quant/convert_qwen3.py)，必须加 `--quant-lm-head`，不加 `--quant-embed-tokens`；[完整命令](#cmd-w8) |
| `text/*/w4a16.toml`，0.6–14B | Text7 W4/G32；**head、embedding、norm FP16**；A16 | 官方 BF16 父 | 加载时；[python/rpu_backend/quant/convert_qwen3_w4a16.py](../../../python/rpu_backend/quant/convert_qwen3_w4a16.py) 的 `convert_qwen3_to_w4a16_`；[运行命令](#w4g32加载时转换head-保留-fp16) |
| `text/*/w4a16_lm_head.toml`，0.6–14B | Text7 **和 head W4/G32**；embedding/norm FP16；A16 | 官方 BF16 父 | 同一加载入口；[运行命令](#w4g32加载时转换-text7--head) |
| `text/32b/w4a16*.toml` | 上述两种 W4 配方由文件名区分 head；A16 | 官方32B BF16父 | 有界、逐层加载时转换；[python/rpu_backend/quant/load_qwen3_quantized.py](../../../python/rpu_backend/quant/load_qwen3_quantized.py) 的 `stage_qwen3_w4a16_for_rpu` / `materialize_staged_qwen3_w4a16_for_rpu`；[完整命令](#cmd-32b-w4) |

W4 不生成离线 checkpoint，原 BF16 目录仍是运行输入；运行时保存的是 packed INT4 与 FP16 scale。W8 会生成独立 checkpoint；源 embedding 与 head 原本共享时，`--quant-lm-head` 会建立独立 INT8 head，embedding 保持浮点。

<a id="quantization"></a>
## 下载与转换

**离线转换可在 CPU 机器完成；各命令块最后一行 `python examples/run.py ...` 是模型推理，需在可用的 设备 板卡上执行。若当前是 CPU 转换机，请只执行下载和转换部分，再将产物按相同缓存路径放到板卡。**

以下 **Bash 命令均在仓库根目录、已安装项目依赖的 Python 环境中执行**。先把 `/absolute/path/to/model_cache` 换成实际缓存根目录：

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache
```

`SRC` 是完整浮点 checkpoint，`DST` 是尚不存在的量化输出目录。已下载权重时可把 `SRC` 改为实际位置并跳过 `hf download`；保持 `DST` 写法，下面对应 TOML 的 `example.registry_alias` 会自动读它，**无需修改 TOML 的模型路径**。此族不支持 `--model`。

<a id="cmd-fp16"></a>
### FP16：下载官方 BF16，加载时转换

```bash
SIZE=4  # 可选：0.6、1.7、4、8
CONFIG_SIZE=${SIZE/./_}
hf download "Qwen/Qwen3-${SIZE}B" --local-dir "$RPU_MODEL_CACHE/Qwen3-${SIZE}B"
python examples/run.py --config "examples/configs/qwen3/text/${CONFIG_SIZE}b/fp16.toml"
```

<a id="cmd-instruct"></a>
### Instruct FP16：使用独立权重

4B 使用已登记的官方 revision：

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir "$RPU_MODEL_CACHE/qwen3-4b-instruct"
python examples/run.py --config examples/configs/qwen3/text/4b/fp16_instruct.toml
```

0.6B 须先将历史已供完整包放到 `$RPU_MODEL_CACHE/qwen3-0.6b-instruct`，再执行：

```bash
python examples/run.py --config examples/configs/qwen3/text/0_6b/fp16_instruct.toml
```

此配置需要原配对包。

<a id="cmd-w8"></a>
### W8：离线转换 Text7 + lm_head，再运行

下面完整块覆盖 0.6/1.7/4/8/14/32B；**只改 `SIZE` 一行选择大小**，输入、输出和模板路径会一起匹配。`--quant-lm-head` 不可省略。

```bash
SIZE=4  # 可选：0.6、1.7、4、8、14、32
SRC="$RPU_MODEL_CACHE/Qwen3-${SIZE}B"
DST="$RPU_MODEL_CACHE/qwen3_lmhead_int8_ckpts/qwen3-${SIZE}b-w8a16-lmhead-int8"
if [ "$SIZE" = 32 ]; then DST="$RPU_MODEL_CACHE/Qwen3-32B-w8a16"; fi
CONFIG_SIZE=${SIZE/./_}
hf download "Qwen/Qwen3-${SIZE}B" --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3.py \
  --src "$SRC" --dst "$DST" --quant-lm-head
python examples/run.py --config "examples/configs/qwen3/text/${CONFIG_SIZE}b/w8a16.toml"
```

脚本：[python/rpu_backend/quant/convert_qwen3.py](../../../python/rpu_backend/quant/convert_qwen3.py)。以 `SIZE=4` 运行时，对应 [text/4b/w8a16.toml](text/4b/w8a16.toml)。新转换不声称重建历史14B已供W8的未知浮点亲缘或原分片字节。

<a id="w4g32加载时转换head-保留-fp16"></a>
### W4/G32：加载时转换，输出头保留 FP16

**没有离线 W4 转换命令，也没有 W4 输出目录。** 以下命令下载4B浮点父，并在公共运行入口冷安装时量化 Text7：

```bash
hf download Qwen/Qwen3-4B --local-dir "$RPU_MODEL_CACHE/Qwen3-4B"
python examples/run.py --config examples/configs/qwen3/text/4b/w4a16.toml
```

<a id="w4g32加载时转换-text7--head"></a>
### W4/G32：加载时转换 Text7 + 输出头

```bash
hf download Qwen/Qwen3-4B --local-dir "$RPU_MODEL_CACHE/Qwen3-4B"
python examples/run.py --config examples/configs/qwen3/text/4b/w4a16_lm_head.toml
```

其他尺寸的两种 W4 模式也直接读取原浮点父。需要切换时复制以下完整块；只改 `SIZE` 和 `MODE`，无需拼接目录或手动调用量化函数：

```bash
SIZE=8  # 可选：0.6、1.7、4、8、14
MODE=w4a16  # head 保留 FP16；改为 w4a16_lm_head 则 head 也量化
CONFIG_SIZE=${SIZE/./_}
hf download "Qwen/Qwen3-${SIZE}B" --local-dir "$RPU_MODEL_CACHE/Qwen3-${SIZE}B"
python examples/run.py --config "examples/configs/qwen3/text/${CONFIG_SIZE}b/${MODE}.toml"
```

<a id="cmd-32b-w4"></a>
### 32B W4/G32：有界加载时转换

32B同样读取浮点父，但必须由以下公共入口逐层量化上传，不能先加载完整CPU FP16模型：

head保留FP16：

```bash
hf download Qwen/Qwen3-32B --local-dir "$RPU_MODEL_CACHE/Qwen3-32B"
python examples/run.py --config examples/configs/qwen3/text/32b/w4a16.toml
```

或选择head也量化：

```bash
hf download Qwen/Qwen3-32B --local-dir "$RPU_MODEL_CACHE/Qwen3-32B"
python examples/run.py --config examples/configs/qwen3/text/32b/w4a16_lm_head.toml
```

FP16模板无需量化，直接使用[权重来源](#weights)的浮点目录；14B/32B浮点父仅用于转换。

## 选择配置

| 模型 | FP16 | W8A16 | W4/G32：FP16 head | W4/G32：W4 head |
|---|---|---|---|---|
| 0.6B | [FP16](text/0_6b/fp16.toml), [Instruct](text/0_6b/fp16_instruct.toml) | [W8A16](text/0_6b/w8a16.toml) | [Text7 W4](text/0_6b/w4a16.toml) | [Text7 + head W4](text/0_6b/w4a16_lm_head.toml) |
| 1.7B | [FP16](text/1_7b/fp16.toml) | [W8A16](text/1_7b/w8a16.toml) | [Text7 W4](text/1_7b/w4a16.toml) | [Text7 + head W4](text/1_7b/w4a16_lm_head.toml) |
| 4B | [FP16](text/4b/fp16.toml), [Instruct](text/4b/fp16_instruct.toml) | [W8A16](text/4b/w8a16.toml) | [Text7 W4](text/4b/w4a16.toml) | [Text7 + head W4](text/4b/w4a16_lm_head.toml) |
| 8B | [FP16](text/8b/fp16.toml) | [W8A16](text/8b/w8a16.toml) | [Text7 W4](text/8b/w4a16.toml) | [Text7 + head W4](text/8b/w4a16_lm_head.toml) |
| 14B | 不支持 | [W8A16](text/14b/w8a16.toml) | [Text7 W4](text/14b/w4a16.toml) | [Text7 + head W4](text/14b/w4a16_lm_head.toml) |
| 32B | 不支持 | [W8A16](text/32b/w8a16.toml) | [Text7 W4](text/32b/w4a16.toml) | [Text7 + head W4](text/32b/w4a16_lm_head.toml) |

FP16 Instruct 使用不同 registry alias；W8 使用已有 INT8 lm_head checkpoint。W4 在冷安装时对浮点资产做对称 G32 量化：`w4a16` 仅七类 decoder Linear，`w4a16_lm_head` 再量化 head；embedding/norm/激活保持 FP16。W4 使用此处的对称量化配方。

各精度的登记支持见 [模型矩阵](../../../docs/model_support.md)。

## 使用

在 `RPU_MODEL_CACHE` 下准备所选 registry alias 对应 checkpoint，复制配置并修改 prompt/输出目录：

```bash
python examples/run.py --config examples/configs/qwen3/text/4b/w4a16_lm_head.toml --check-config
python examples/run.py --config /path/to/my_qwen3.toml
```

FP16 使用浮点 checkpoint；W8 需量化 decoder、独立量化 head 和浮点 embedding。运行需空闲板卡及既有设备锁，冷配置变化后使用新进程。输出生成结果；`--check-config` 仅检查配置。

采样设置示例为 `run.warmup=2`、`run.runs=5`；模板默认次数见所选文件，保持对应输入、精度和核数。

## 可修改参数

| 字段 | 默认 / 可选值 | 条件 |
|---|---|---|
| `input.prompt`、`prefill_tokens`、`decode_steps` | 依模板 | decode_steps为首token之后的decode上限，实际输出≤上限+1 |
| `rpu_execution.model.num_cores` | FP16 B1：4/6/8，默认8 | 量化模板保持TP8；缩核不继承TP8证据 |
| `rpu_execution.prefill.linear_acc32` | false=ACC16 / true=ACC32 | 支持此字段的精确profile，安装前设置 |
| `rpu_execution.prefill.chunk_size` | auto | 受模型资源范围限制 |
| `cpu.threads`、`run.warmup/runs` | 依模板 | 改条件后不继承旧计时 |

## 输入与限制

14B 量化保留 P192/C32 envelope；32B 量化 KV上限4096、chunk≤64，请求仍由运行时准入检查。32B W4必须直接 `device="rpu"` 有界加载，禁止先物化全CPU FP16模型；32B W8使用量化 checkpoint mmap加载。FP16 14B/32B与量化缩核不支持。

库依赖与资产要求见 [正式 main/DDR 两份 REF](../../../docs/runtime_assets.md)、[量化范围](../../../docs/quantization.md)。
