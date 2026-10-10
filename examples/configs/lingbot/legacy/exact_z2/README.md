# LingBot v2 exact-Z2 归档

[6b_w8a16.toml](6b_w8a16.toml) 保留 `lingbot-vla-v2-6b.w8a16.exact-z2` 配置，原路径为 `lingbot/v2/6b_w8a16.toml`。新集成使用 [v2 runtime Graph](../../v2/README.md)。

此配置通过 `Lingbot2Policy.prepare_graphs` / `infer` 运行，输入为三相机 256×256、有效 prefix 215–225、固定执行 P225、W8A16 和 true/true 掩码，不接受 RTC 动作或延迟参数。

<a id="weights"></a>
<a id="weight-sources"></a>
## 官方来源与历史父资产

| 资产 | 官方下载 / 源码 | 原始存储精度与边界 |
|---|---|---|
| 官方 v2 6B 原始父权重 | [HF](https://huggingface.co/robbyant/lingbot-vla-v2-6b)；[GitHub](https://github.com/robbyant/lingbot-vla-v2) | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` 的主要模型 safetensors 为 **F32**，见 [HF 元数据](https://huggingface.co/api/models/robbyant/lingbot-vla-v2-6b) |
| 旧 BF16 交付父权重 | 原配对交付 | **BF16 衍生资产**，不是上述官方下载；必须保留原版本、哈希、源码/训练配置和输入，不能与 F32 参考互换 |

下面只能取得官方 F32 父权重，适用于匹配的原始 F32 记录；不能复现绑定另一 BF16 交付资产的结果。旧 registry 的目录层级与当前 v2 不同：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download robbyant/lingbot-vla-v2-6b --revision 11c703bf6a5c1f45b3b69168482da11fdbba53d7 --local-dir "$RPU_MODEL_CACHE/hf/robbyant/lingbot-vla-v2-6b"
```

<a id="weight-recipes"></a>
## 模块精度与配方

| 配置 | 模块实际 W/A 精度 | 输入父权重 → 转换方式与命令 |
|---|---|---|
| `6b_w8a16.toml` | Vision block 六类投影、Text 七类投影、Action 注意力及共享/路由专家三类 MLP 投影为 W8/A16（INT8 + FP16 scale）；Vision patch/merger、Action router/IO/条件与其他浮点参数不量化 | 对应历史记录的父资产 → [旧加载时量化完整命令](#quantization)，经 `Lingbot2Policy` → `build_lingbot_vla_v2()` / `build_expert_moe()`；没有离线转换 CLI 或独立量化下载 |

这里只保留 exact-Z2、grouped-expert 和 true/true 掩码的历史合同，不将当前 v2 稀疏路径替换进来，也不新增支持声明。

<a id="quantization"></a>
## 加载时量化：归档配置这样运行

以下命令在仓库根目录执行。量化在最后一行加载模型时发生，不生成独立量化目录。先替换实际路径，再按注释用 `vi` 编辑副本并保存；`--check-config` 只校验配置，不能代替最后的运行命令。

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
# 此旧 registry 需要 hf/robbyant/lingbot-vla-v2-6b 子目录及原交付配套源码/配置。
cp examples/configs/lingbot/legacy/exact_z2/6b_w8a16.toml /tmp/lingbot2_exact_z2_w8.toml
# input.observation = 匹配原 exact-Z2 的真实 observation 文件。
# 在 [run] 中添加 output_dir = 本次目录；保留原 profile、dtype 和受控 opt-in。
vi /tmp/lingbot2_exact_z2_w8.toml
python examples/run.py --config /tmp/lingbot2_exact_z2_w8.toml
```
这里使用旧加载期 W8 安装，不导出量化目录；原父权重、true/true 掩码与固定输入合同不能用当前 v2 配方替换。

**实现位置**：[python/rpu_backend/adapters/lingbot_vla_v2/runtime.py](../../../../../python/rpu_backend/adapters/lingbot_vla_v2/runtime.py)；[python/rpu_backend/adapters/lingbot_vla_v2/convert.py](../../../../../python/rpu_backend/adapters/lingbot_vla_v2/convert.py)。旧 `Lingbot2Policy` 加载路径调用 `build_lingbot_vla_v2()` / `build_expert_moe()`；实现文件不是 CLI。

## 权重与运行

官方原始 F32 权重与下载版本见 [v2 权重说明](../../v2/README.md#weights)。归档入口在加载时生成 W8；已有 BF16 派生资产须配套使用原训练配置和输入。

复制模板，填写 checkpoint、配套源码和含显式噪声的 CPU observation，再执行：

```bash
python examples/run.py --config /path/to/exact_z2.toml --check-config
python examples/run.py --config /path/to/exact_z2.toml
```

`input.prefix_length_range` 默认 `[215,225]`；`run.warmup/runs/seed` 默认 `2/3/0`；`cpu.threads` 可设正整数。保留 `RPU_LINGBOT2_ALLOW_UNVALIDATED=1` 和 `RPU_LINGBOT2_MULTIVIEW_SPM_Z2=1`；改变冷配置后启动新进程。
