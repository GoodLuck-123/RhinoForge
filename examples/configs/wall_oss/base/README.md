<a id="wall-oss-base-配置"></a>

# Wall-OSS 基础配置

本组使用公开 Wall-OSS 0.5。按两/三图、224/256 输入选择目录，再选精度；固定 batch1、TP8、动作长度 32、十步去噪。两图为 face/left wrist，三图增加 right wrist。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

## 配置说明

| 配方 | 两图 224 | 两图 256 | 三图 224 | 三图 256 | 权重获取 |
|---|---|---|---|---|---|
| FP16 | [配置](2cam/224/fp16.toml) | [配置](2cam/256/fp16.toml) | [配置](3cam/224/fp16.toml) | [配置](3cam/256/fp16.toml) | [原始公开模型](#download-parent) |
| W8A16 | [配置](2cam/224/w8a16.toml) | [配置](2cam/256/w8a16.toml) | [配置](3cam/224/w8a16.toml) | [配置](3cam/256/w8a16.toml) | [原始模型 + W8 目录](#公开-base-w8a16) |
| W8 + 动作 NVFP4 | [配置](2cam/224/w8a16_action_nvfp4.toml) | [配置](2cam/256/w8a16_action_nvfp4.toml) | [配置](3cam/224/w8a16_action_nvfp4.toml) | [配置](3cam/256/w8a16_action_nvfp4.toml) | [复用原始/W8 目录](#公开-base-w8--action-nvfp4) |

<a id="fp16-基准与配置增量"></a>

### FP16 基准

以相同相机数、分辨率的 `fp16.toml` 为基准。Vision、Language expert0、Action expert1 主体投影使用 FP16，三个阶段默认 `linear_acc32=false`，即 ACC16。

基准已有设备 patch embedding、预处理归一化折叠、图像与文本输入缓存、设备 embedding、融合输入组装、merger 融合、部分 MRoPE、KV 对齐、十步去噪展开与融合，以及 Graph replay。`vision_row_reuse=true` 开启视觉行复用；FP16 还选择对应的浮点传输路径。默认 Action ACC16 使用权重驻留，减少多步去噪的重复搬运。

224 输入设 `vision_window=true`、`vision_pair=false`，采用窗口 attention；256 输入经 processor 变为 252×252，设 `vision_window=false`、`vision_pair=true`，前两图合批，三图的第三路单独执行。这两组均复用同精度权重。

### W8A16：三个组件的主投影转 INT8

`wall_oss.precision` 从 `"fp16"` 改为 `"w8a16"`，新增 `input.w8_checkpoint`。Vision block、Language expert0 和 Action expert1 的 Q/K/V/O/Gate/Up/Down 使用 W8A16：逐输出通道 INT8 权重、FP16 scale 和 FP16 激活。公共缓存、融合与相机执行方式继续使用，FP16 专用传输路径改为量化权重的路径。

两种量化配方都需要原始目录：patch/merger、embedding、norm/bias、动作预处理和输出回投继续使用浮点，设备部分为 FP16，CPU 归一化为 FP32。W8 转换器会量化除 `embed_tokens` 外的浮点二维 `.weight`；运行器按模块选择，动作预处理仍读取原始浮点权重。因此离线文件中的 INT8 范围大于实际运行的 INT8 范围。

### 动作 NVFP4：保留 Vision/Language W8，量化完整动作专家

在 W8 上将 `wall_oss.precision` 改为 `"w8a16_action_nvfp4"`，两项权重路径不变。加载时从原始浮点目录转换 Action expert1 的 Q/K/V/O/Gate/Up/Down，包含 K/V，使用 E2M1 四位浮点、K 维 block16 的 E4M3FN scale 和 FP32 tensor scale。激活仍为 FP16。

该配方不启用 FP16/W8 使用的 Action 权重驻留。因此它同时改变权重格式和动作执行路径，没有独立 NVFP4 导出目录。

### 模板中的线程和 padding 差异

各模板还带有对应 T32 测试输入的 CPU 线程数和预填充 padding。下表每格为“线程数 / `padding_rows`”；“自动”表示 TOML 未设置固定行数。

| 输入 | FP16 | W8A16 | W8 + 动作 NVFP4 |
|---|---|---|---|
| 两图 224 | 4 / 自动 | 8 / 自动 | 4 / 60 |
| 两图 256 | 4 / 26 | 8 / 26 | 8 / 26 |
| 三图 224 | 4 / 自动 | 12 / 33 | 8 / 33 |
| 三图 256 | 8 / 自动 | 12 / 自动 | 12 / 自动 |

padding 只改变物理计算行数，不增加有效 token。更换指令或输入后先删除固定 `padding_rows/padding_budget`，使用自动规划。

实现见[配置入口](../../../runners/wall_oss/base.py)、[执行参数选择](../../../runners/wall_oss/base_engine.py)、[W8 转换器](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)及[动作加载器](../../../../python/rpu_backend/adapters/wall_oss/action.py)。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

## 准备与运行

### 获取所选权重

源模型为 X Square Robot 的 [wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028)，下载命令固定 revision `b7dc23a82dbf70a65859c9fdd596703234597028`。发布文件含 BF16 和 FP32，保留完整 checkpoint、processor/tokenizer 与 normalizer；源码见 [wall-x](https://github.com/X-Square-Robot/wall-x)。

以下在仓库根目录执行，原始目录与 W8 目录放在同一父目录，便于解析转换元数据。

<a id="download-parent"></a>

<a id="run-fp16"></a>

**FP16：下载原始模型。**

```bash
WALL_SRC=/absolute/path/model-cache/wall-oss-0.5
hf download x-square-robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir "$WALL_SRC"
```

<a id="公开-base-w8a16"></a>

<a id="公开-base-w8--action-nvfp4"></a>

**W8 和动作 NVFP4：再准备同一套 W8 目录。**

```bash
WALL_W8=/absolute/path/model-cache/wall-oss-0.5-w8a16
python python/rpu_backend/quant/convert_wall_oss_w8a16.py \
  --src "$WALL_SRC" --dst "$WALL_W8"
```

W8 配方和动作 NVFP4 配方都填写 `input.fp16_checkpoint=WALL_SRC 的实际路径`、`input.w8_checkpoint=WALL_W8 的实际路径`。FP16 只填写前者；动作 NVFP4 没有独立离线目录。

<a id="使用"></a>

<a id="运行与输入"></a>

### 绑定输入并运行

从第一章选择 TOML，下面以三图 224 W8 为例：

```bash
WALL_CONFIG=examples/configs/wall_oss/base/3cam/224/w8a16.toml
cp "$WALL_CONFIG" /tmp/wall_base.toml
# input.fp16_checkpoint / input.w8_checkpoint = 上面的实际目录。
# input.batch_template = 实际 batch 路径模板；run.output_dir = 本次结果目录。
# 输入与模板原测试指令不同时，移除固定 padding_rows/padding_budget，使用自动规划。
vi /tmp/wall_base.toml
python examples/run.py --config /tmp/wall_base.toml
```

<a id="输入与输出"></a>

### 输入格式

`input.batch` 指向一个已保存的 CPU 字典；也可使用互斥的 `batch_template`，展开 `{cameras}`、`{image_size}`、`{text_tokens}`，其中必须包含 `{text_tokens}`。

batch 包含六项：uint8 `images [N,H,W,3]`（face/left/right 的前 N 路），非空 `instruction`，有限浮点 `proprioception [26]`、`noise [1,32,26]`、`dof_mask [1,32,26]`，以及 bool `state_mask [26]`。该输入使用 `berkeley_autolab_ur5` 前七个槽位，state mask 前 7 项为 true，dof mask 扩展相同七维选择。

<a id="输入与执行变体"></a>

### 常调参数

| 字段 | 模板值或入口默认 | 可选范围 | 修改时的关联 |
|---|---|---|---|
| `input.cameras/image_size` | 2/3 图，224/256 | 2 / 3；`[224,224]` / `[256,256]` | batch 与实际几何一致；同精度权重复用 |
| `input.text_tokens` | 32 | 正整数；与运行器准入匹配 | 与 instruction 实际 token 数一致 |
| `input.text_length_scope` | instruction | instruction / non_image | 后者统计完整 prompt 的非图像 token |
| `rpu_execution.prefill/vision/action.linear_acc32` | false | false / true | 三阶段分别选择；Action ACC32 停用 ACC16 配方的权重驻留 |
| `rpu_execution.prefill.chunk_size` | auto | 自动或有效整数 | 固定 padding 与完整 prompt 绑定 |
| `rpu_execution.action.chunk_size` | 未写，auto | auto / 32 | 固定动作长度 32 |
| `wall_oss.vision_window` | 224 为 true，256 为 false | false / true | true 要求 224 且开启 row reuse |
| `wall_oss.vision_pair` | 224 为 false，256 为 true | false / true | true 要求 256；三图第三路单独执行 |
| `wall_oss.vision_row_reuse` | true | false / true | 改变视觉行复用及传输路径 |
| `wall_oss.instruction_residency` | 模板省略 | auto / off | 快照重放选择，与 Action 权重驻留分别控制 |
| `cpu.threads` | 见第一章线程表；省略默认 8 | 正整数 | CPU 线程与权重精度独立 |
| `run.warmup/runs/seed` | 2 / 5 / 0 | 预热 1–10；采样 1–30；seed 非负 | seed 不替换 batch 中已有 noise |

公共字段见[参数说明](../../PARAMETERS.md)。运行使用[正式 main/DDR 两份 REF](../../../../docs/runtime_assets.md)，`[runtime]` 可指定库路径；更换冷配置后用新进程。

## 输出

`run.output_dir` 保存 `report.json` 和 `outputs.pt`，分别记录有效配置、输入身份和动作。
