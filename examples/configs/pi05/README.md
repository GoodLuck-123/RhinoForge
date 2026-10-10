# Pi0.5 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；推荐 [3cam/w8a16.toml](3cam/w8a16.toml)。固定 B1、horizon50、十步去噪。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方发布 / 当前实际父 | LeRobot：[pi05_libero_finetuned_v044（LIBERO 微调）](https://huggingface.co/lerobot/pi05_libero_finetuned_v044/tree/dbf8a3f794a9c4297b44f40b752712f50073d945) | BF16 + FP32（官方 safetensors 元数据） | `pi05_libero_finetuned`；原始配置的 `input.checkpoint`，见[下载](#download-parent) |
| 本仓库转换资产 | 由下列转换器从该 LIBERO 父生成；使用本仓库转换配方 | W8 文件：Gemma 投影 INT8，其余浮点 FP16；NVFP4 文件：动作五类投影 E2M1 + FP8/FP32 scale，K/V 与 VLM INT8；Vision/AdaRMS/IO/norm 等其余浮点仍为 FP16 | `pi05_libero_finetuned-w8a16-vlm-expert` 或 `pi05_action_nvfp4` → `input.checkpoint` |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../WEIGHTS.md#precision-notation)。 NVFP4A16 为 E2M1 四位权重、沿 K 每 16 个值一组的 E4M3FN 八位 scale、FP32 tensor scale，激活仍为 FP16； 相机数、长度和 TP4/6 不新增离线量化配方。

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| FP16：[两图](2cam/fp16.toml) / [三图](3cam/fp16.toml) / [TP4](unoptimized/3cam/fp16_tp4.toml) / [TP6](unoptimized/3cam/fp16_tp6.toml) | Vision、Gemma VLM 投影 W16A16 | 动作投影及 AdaRMS W16A16；无整数权重量化 | 原始 LIBERO 父；直接加载；[完整命令](#run-fp16) |
| W8A16：[两图](2cam/w8a16.toml) / [三图](3cam/w8a16.toml) / [TP4](unoptimized/3cam/w8a16_tp4.toml) / [TP6](unoptimized/3cam/w8a16_tp6.toml) | SigLIP Q/K/V/O/FC1/FC2、VLM Q/K/V/O/Gate/Up/Down 为 W8A16 | 动作 Q/K/V/O/Gate/Up/Down 为 W8A16；patch embedding、位置、归一化参数/偏置、多模态投影及动作输入/输出投影保留 FP16；AdaRMS dense 单独 W8A16（加载时），其偏置仍浮点 | 离线文件中 VLM/Action 投影 INT8，SigLIP/AdaRMS 仍 FP16；加载时再把其指定投影转 W8；[W8 命令](#run-w8)，[python/rpu_backend/quant/convert_pi05.py](../../../python/rpu_backend/quant/convert_pi05.py) |
| 动作 NVFP4：[两图](2cam/w8a16_action_nvfp4.toml) / [三图](3cam/w8a16_action_nvfp4.toml) | 同 W8A16 | 动作 Q/O/Gate/Up/Down 为 NVFP4A16（`striped_v2`）；K/V 为 W8A16；patch embedding、位置、归一化参数/偏置、多模态投影及动作输入/输出投影保留 FP16；AdaRMS dense 单独 W8A16（加载时），其偏置仍浮点 | 原始父 → 离线动作五类 NVFP4 + K/V/VLM W8；文件中 SigLIP/AdaRMS 为 FP16，加载时再转 W8；[完整命令](#w8--action-nvfp4离线转换后运行) |
| 预填充 W8A8 + 动作 NVFP4：[两图](2cam/w8a16_prefill_w8a8_action_nvfp4.toml) / [三图](3cam/w8a16_prefill_w8a8_action_nvfp4.toml) | 同上，仅 VLM 预填充 Gate/Up 的输入激活额外量化为 INT8（W8A8），其余为 A16 | 同动作 NVFP4 配方；IO/归一化不变 | 共用动作 NVFP4 目录，TOML 启用预填充 A8；[完整命令](#w8--prefill-w8a8--action-nvfp4) |

<a id="quantization"></a>
## 下载与转换

以下命令在仓库根目录执行。先把 `/path/to/...` 换成实际路径；`vi` 这一步编辑刚复制的 TOML，保存后再执行下一行。每个代码块是一个独立精度方案，只执行所需方案；已有同身份量化目录时跳过对应转换行。

<a id="download-parent"></a>
### 下载原始 LIBERO 父权重

```bash
SRC=/path/to/model-cache/pi05_libero_finetuned
hf download lerobot/pi05_libero_finetuned_v044 --revision dbf8a3f794a9c4297b44f40b752712f50073d945 --local-dir "$SRC"
```

<a id="run-fp16"></a>
### FP16：原始浮点父权重直接运行

```bash
cp examples/configs/pi05/3cam/fp16.toml /tmp/pi05_fp16.toml
# input.checkpoint = 下载得到的 pi05_libero_finetuned 绝对目录。
# input.batch = 匹配三图 T32 的真实 batch；run.output_dir = 本次独立目录。
vi /tmp/pi05_fp16.toml
python examples/run.py --config /tmp/pi05_fp16.toml
```

<a id="run-w8"></a>
### W8A16：离线转换后运行

```bash
SRC=/path/to/model-cache/pi05_libero_finetuned  # 原始未量化 LIBERO 父目录
DST=/path/to/model-cache/pi05_libero_finetuned-w8a16-vlm-expert  # 新输出目录
python python/rpu_backend/quant/convert_pi05.py --src "$SRC" --dst "$DST"
cp examples/configs/pi05/3cam/w8a16.toml /tmp/pi05_w8.toml
# input.checkpoint = 上面 DST 的实际绝对路径；input.batch = 匹配三图 T32 的真实 batch 文件。
# run.output_dir = 本次独立输出目录。保留模板 precision。
vi /tmp/pi05_w8.toml
python examples/run.py --config /tmp/pi05_w8.toml
```

<a id="w8--action-nvfp4离线转换后运行"></a>
### W8 + 动作 NVFP4：离线转换后运行

```bash
SRC=/path/to/model-cache/pi05_libero_finetuned  # 原始未量化 LIBERO 父目录
DST=/path/to/model-cache/pi05_action_nvfp4  # 新输出目录
python python/rpu_backend/quant/convert_pi05.py --src "$SRC" --dst "$DST" --action-w4 --action-w4-format nvfp4
cp examples/configs/pi05/3cam/w8a16_action_nvfp4.toml /tmp/pi05_nvfp4.toml
# input.checkpoint = 上面 DST 的实际绝对路径；input.batch = 匹配三图 T32 的真实 batch 文件。
# run.output_dir = 本次独立输出目录。保留模板 precision。
vi /tmp/pi05_nvfp4.toml
python examples/run.py --config /tmp/pi05_nvfp4.toml
```

<a id="w8--prefill-w8a8--action-nvfp4"></a>
### W8 + 预填充 W8A8 + 动作 NVFP4

```bash
SRC=/path/to/model-cache/pi05_libero_finetuned  # 原始未量化 LIBERO 父目录
DST=/path/to/model-cache/pi05_action_nvfp4  # 新输出目录
python python/rpu_backend/quant/convert_pi05.py --src "$SRC" --dst "$DST" --action-w4 --action-w4-format nvfp4
cp examples/configs/pi05/3cam/w8a16_prefill_w8a8_action_nvfp4.toml /tmp/pi05_w8a8_nvfp4.toml
# input.checkpoint = 上面 DST 的实际绝对路径；input.batch = 匹配三图 T32 的真实 batch 文件。
# run.output_dir = 本次独立输出目录。保留模板 precision。
vi /tmp/pi05_w8a8_nvfp4.toml
python examples/run.py --config /tmp/pi05_w8a8_nvfp4.toml
```

两图选择 `2cam/` 下同精度模板；W8 的 TP4/6 模板使用同一 W8 输出。Prefill W8A8 与普通 Action NVFP4 共用离线产物。**FP16 无需量化**，使用原始父目录。

**实现位置**：[python/rpu_backend/quant/convert_pi05.py](../../../python/rpu_backend/quant/convert_pi05.py)；[examples/runners/pi05/optimized.py](../../runners/pi05/optimized.py)；[python/rpu_backend/adapters/pi05/__init__.py](../../../python/rpu_backend/adapters/pi05/__init__.py)；[python/rpu_backend/adapters/siglip.py](../../../python/rpu_backend/adapters/siglip.py)。离线 `main()` 调用 `convert_checkpoint()`；加载入口 `run_profile()` 经 `Pi05Adapter.to_rpu()` / `_install_fused_denoise_handle()` 安装 SigLIP/AdaRMS W8，SigLIP 调用 `_convert_siglip_weights_for_rpu()`。旧 INT4-G128 与 fake-W4 不能替代 NVFP4。

## 选择配置

| 权重 / 激活组合 | 两图 TP8 | 三图 TP8 | 区别 |
|---|---|---|---|
| FP16 | [配置](2cam/fp16.toml) | [配置](3cam/fp16.toml) | 浮点权重 |
| W8A16 | [配置](2cam/w8a16.toml) | [配置](3cam/w8a16.toml) | VLM 与 Action INT8 权重 |
| W8 + Action NVFP4 | [配置](2cam/w8a16_action_nvfp4.toml) | [配置](3cam/w8a16_action_nvfp4.toml) | Action Q/O/Gate/Up/Down NVFP4；K/V 与 VLM W8 |
| W8 + Prefill GateUp W8A8 + Action NVFP4 | [配置](2cam/w8a16_prefill_w8a8_action_nvfp4.toml) | [配置](3cam/w8a16_prefill_w8a8_action_nvfp4.toml) | 仅 Prefill GateUp 加激活 INT8 |

三图未优化通用路径另有 [FP16 TP4](unoptimized/3cam/fp16_tp4.toml) / [TP6](unoptimized/3cam/fp16_tp6.toml)、[W8 TP4](unoptimized/3cam/w8a16_tp4.toml) / [TP6](unoptimized/3cam/w8a16_tp6.toml)，仅 T32。两图、NVFP4、长文本及自定义资产不继承三图 T32 登记状态；具体 profile 见 [模型矩阵](../../../docs/model_support.md)。

## 使用

复制模板，替换 `input.checkpoint`、真实 `batch` 和 `run.output_dir`。模型缓存通过 `RPU_MODEL_CACHE`/registry alias 定位；NVFP4 必须使用配对 `striped_v2/block16` checkpoint，旧 INT4 资产不可替代。

```bash
cp examples/configs/pi05/3cam/w8a16.toml /tmp/pi05.toml
python examples/run.py --config /tmp/pi05.toml --check-config
python examples/run.py --config /tmp/pi05.toml
```

若需要转换 Action NVFP4，从原 FP16 模型执行：

```bash
python -m rpu_backend.quant.convert_pi05 --src /path/to/pi05_fp16 --dst /path/to/pi05_action_nvfp4 --action-w4
```

`--action-w4` 默认采用 NVFP4，参数见 `--help`。运行需匹配 native、空闲板卡和既有设备锁；冷配置变化后使用新进程。输出 `report.json`、`actions.pt`；`--check-config` 仅检查配置。

采样设置示例：`cpu.threads=8`、`run.warmup=2`、`run.runs=5`；模板默认6次正式运行。

## 可修改参数

| 字段 | 可选值 / 默认 | 条件 |
|---|---|---|
| `input.text_tokens` | TP8：32/64/96/128/160；默认32 | 同时提供真实 token 与 mask |
| `rpu_execution.model.num_cores` | 8；另有 TP4/6 模板 | NVFP4/A8 要求 TP8 与 paired 布局 |
| `rpu_execution.prefill/vision/action.linear_acc32` | false=ACC16，true=ACC32 | 分组件冷绑定，不改变权重格式 |
| `rpu_execution.prefill.chunk_size` | auto | 自定义 chunk 不继承精确 paired profile；NVFP4/A8 不匹配时拒绝 |
| `cpu.threads` | 8，正整数 | CPU intra-op/OMP/MKL 线程；与设备核数无关 |
| `run.warmup/runs/seed` | 2 / 6 / 0 | 可用 `input.noise` 固定 CPU noise |
| `run.profile/hwperf` | false | 诊断调用和跟踪不能作为无跟踪计时 |

AUTO paired chunk 对应 T32/64/96/128/160：两图 C272/288/304/320/336，三图 C400/416/432/448/464。

## 输入与限制

`batch` 是 CPU tensor 字典：每个 `observation.images.*` 槽位为 `[1,3,256,256]`，须与 checkpoint 配置一致。两图须使用两路真实相机，不含 `empty_camera_*`；三槽可保留 checkpoint 定义的占位槽，所有配置槽位均须提供。`observation.language.tokens` 为整数 `[1,T]`，attention mask 为同宽非空 bool 右 padding。入口不会生成或截断文本，相机槽位必须满足 checkpoint 配置；存在图像 padding mask 时必须为 `[1]` bool true。其他字段遵循 `Pi05Policy.predict_action_chunk`。

普通入口不运行 CPU 参考。库依赖见 [正式 main/DDR 两份 REF](../../../docs/runtime_assets.md)。
