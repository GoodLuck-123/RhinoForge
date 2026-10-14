# LingBot v2 exact-Z2 配置

[模型入口](../../README.md) · [runtime Graph 配置](../../v2/README.md) · [w8a16.toml](6b/w8a16.toml) 对应 `Lingbot2Policy.prepare_graphs()` / `infer()` 入口，使用三相机 256×256、有效前缀 215–225、固定物理前缀 P225、true/true 掩码与 W8A16。该配置用于匹配 exact-Z2 输入。

## 配置说明

<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

| 配置 | 精度与执行范围 | 权重获取 |
|---|---|---|
| [w8a16.toml](6b/w8a16.toml) | W8A16，exact-Z2，多视角 SPM，固定 P225 | [对应记录的 FP32 或 BF16 父资产](#weights)，加载时量化 |

模板绑定 `example.profile_id="lingbot-vla-v2-6b.w8a16.exact-z2"`，`example.dtype="w8a16"`；量化在加载时完成，不另存权重文件。

<a id="本配方做了什么"></a>

### 模块精度与执行

| 模块 | W8A16 范围 | 浮点部分 |
|---|---|---|
| Vision | block 的 Q/K/V/O、FC1/FC2 | patch embedding、merger、norm/bias |
| Text | Q/K/V/O、Gate/Up/Down | embedding、norm/bias |
| Action | Q/K/V/O，共享与路由专家 Gate/Up/Down | router、IO、条件投影、norm/bias |

W8 使用 INT8 权重、FP16 scale 与激活。执行采用 exact-Z2 多视角 SPM 与 grouped-expert 路径，保持 36 层 Action、32 个路由专家加共享专家、Top4 和原生十步去噪。

本目录只有这一种精度配置。另一套 [runtime Graph 配置](../../v2/README.md)提供 FP16/W8A16 与两/三相机输入，使用训练 YAML 掩码及稀疏专家路径。两套入口的掩码、执行布局和计时定义见[模型入口对照表](../../README.md#v2-两种入口的区别)。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="权重准备"></a>

### 获取配对权重

`example.registry_alias="lingbot-vla-v2-6b"` 读取 `$RPU_MODEL_CACHE/hf/robbyant/lingbot-vla-v2-6b`。官方 FP32 与交付 BF16 是两套父资产，按所复现记录的权重、输入和掩码选择。

| 来源 | 版本与所需配套资产 |
|---|---|
| 官方 [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b) | FP32，revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7`，配套 [官方源码](https://github.com/robbyant/lingbot-vla-v2) |
| BF16 交付资产 | 使用原交付包、源码/配置及其文件校验记录 |

复现 FP32 父权重记录时，下载到本入口的 registry 目录：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download robbyant/lingbot-vla-v2-6b \
  --revision 11c703bf6a5c1f45b3b69168482da11fdbba53d7 \
  --local-dir "$RPU_MODEL_CACHE/hf/robbyant/lingbot-vla-v2-6b"
```

BF16 配方使用原 BF16 交付文件。

<a id="quantization"></a>
<a id="使用"></a>

### 输入与运行

在已安装项目及模型依赖的设备环境、仓库根目录运行：

```bash
cp examples/configs/lingbot/legacy/exact_z2/6b/w8a16.toml /tmp/lingbot2_exact_z2_w8.toml
# input.observation 填匹配 exact-Z2 的 CPU 张量文件。
# 在 [run] 中添加 output_dir。
vi /tmp/lingbot2_exact_z2_w8.toml
python examples/run.py --config /tmp/lingbot2_exact_z2_w8.toml --check-config
python examples/run.py --config /tmp/lingbot2_exact_z2_w8.toml
```

公共入口先 `prepare_graphs()`，然后重复 `infer()`。它使用常规推理接口，配置中没有 RTC 的历史动作或延迟参数。

<a id="输入与执行参数"></a>

### 常用参数

| 字段 | 模板值 | 省略时 | 范围与关联 |
|---|---|---|---|
| `input.observation` | 文件占位路径 | 必填 | 配对 CPU observation，含初始噪声，使用 true/true 掩码 |
| `input.prefix_length_range` | `[215,225]` | `[215,225]` | 有效前缀范围，物理执行为 P225 |
| `input.seed` | 0 | 0 | observation 已提供显式初始噪声 |
| `example.opt_in.RPU_LINGBOT2_ALLOW_UNVALIDATED` | `"1"` | 必填 | 选择受控入口 |
| `example.opt_in.RPU_LINGBOT2_MULTIVIEW_SPM_Z2` | `"1"` | 必填 | 选择 exact-Z2 布局 |

模板 `run.warmup/runs/seed=2/3/0`，`cpu.threads` 未配置。通用字段的省略值、输出与诊断选项见[公共参数说明](../../../PARAMETERS.md)。

加载与调用见[runner](../../../../runners/lingbot2/runner.py)，量化实现见 [runtime.py](../../../../../python/rpu_backend/adapters/lingbot_vla_v2/runtime.py) 和 [convert.py](../../../../../python/rpu_backend/adapters/lingbot_vla_v2/convert.py)。

## 输出

`run.output_dir` 保存 `report.json` 和完整 55 维归一化动作快照。
