# Hy-Embodied UMI 登记配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`examples/run.py --config <file>`。这是原登记的UMI FP16输入接口；优化输入另见 [HyVLA三图模板](../hy_vla/3cam/fp16.toml)。运行时显式选择本页 UMI 配置。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方发布 / 当前实际父 | 腾讯：[Hy-Embodied-0.5-VLA-UMI](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI/tree/3f53d1f8d2bc587c523cfdc9f1041ceee42c2524) | BF16（官方 safetensors 元数据） | `$RPU_MODEL_CACHE/Hy-Embodied-0.5-VLA-UMI` → 模板 registry 别名；目录名区分大小写 |
| 公开量化 | 本登记页不使用量化资产；W8/NVFP4 仅在独立 HyVLA 模板内加载时生成 | 不适用 | 本页只用原始 UMI 目录，不接收 HyVLA 内存量化产物 |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../WEIGHTS.md#precision-notation)。

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| [umi/fp16.toml](umi/fp16.toml) | ViT / VLM 两分支主投影 W16A16 | 动作 `_v` 主投影与图内 IO W16A16；patch/merger/norm/bias FP16；host embedding、state/时间边界 FP32 | 官方 BF16 父；加载时按运行模块转浮点 dtype，不做整数/FP4 量化；[下载与运行](#run-fp16)，[examples/runners/hy_vla/runner.py](../../../examples/runners/hy_vla/runner.py) |

<a id="quantization"></a>
## 下载与运行

<a id="run-fp16"></a>
```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download tencent/Hy-Embodied-0.5-VLA-UMI --revision 3f53d1f8d2bc587c523cfdc9f1041ceee42c2524 --local-dir "$RPU_MODEL_CACHE/Hy-Embodied-0.5-VLA-UMI"
cp examples/configs/hy_embodied/umi/fp16.toml /tmp/hy_umi_fp16.toml
# input.images = 按 head/left/right 排序的三图；input.instruction/state = 配对真实输入。
# 在 [run] 添加本次 output_dir；保留 profile_id、registry_alias、dtype 与 prefix_len。
vi /tmp/hy_umi_fp16.toml
python examples/run.py --config /tmp/hy_umi_fp16.toml
```

**无需量化转换。** 本目录只有 `umi/fp16.toml`，读取原始 UMI 浮点权重，不执行离线量化。不要在这个 registry 模板中改精度标签；W8/NVFP4 属于独立 HyVLA 配方。

## 选择配置

| 配置 | 范围 / 精度 | 登记状态 |
|---|---|---|
| [umi/fp16.toml](umi/fp16.toml) | 三图、UMI state、FP16、P240 | SUPPORTED_LIMITED，immutable v2 profile |

同资产的两图/量化/多文本测试在 [HyVLA目录](../hy_vla/README.md)，不继承本登记fixture的资格。

## 使用

通过 `RPU_MODEL_CACHE` 准备 `hy-embodied-0.5-vla-umi`，填写head/left/right三张图片、真实instruction与state：

```bash
python examples/run.py --config examples/configs/hy_embodied/umi/fp16.toml --check-config
python examples/run.py --config /path/to/hy_umi.toml
```

输出为归一化动作，公共runner保存report与输出快照；不是物理机器人执行。

## 可修改参数

| 字段 | 默认 / 条件 |
|---|---|
| `input.images` | 恰好三图、保持相机顺序 |
| `input.instruction/state` | 模型processor/tokenizer与原state范围 |
| `input.noise_seed` | 0 |
| `input.prefix_len` | 240；原精确profile限制 |
| `cpu.threads` | 可选正整数；省略继承环境 |
| `run.warmup/runs/seed` | 2 / 3 / 0 |

## 输入与限制

保留原FP16、固定前缀和输入语义；修改输入须满足原profile，不能用优化模板字段混入registry schema。支持范围见 [模型矩阵](../../../docs/model_support.md)。

运行环境与结果说明见[示例指南](../../README.zh.md)，运行库准备见[算子库规则](../../../docs/runtime_assets.md)。
