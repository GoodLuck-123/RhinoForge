<a id="已归档legacy-qwen38-27b-w8a16-配置"></a>

# Qwen3.8-27B 配套权重配置

/ [Qwen3.5 配置](../../README.md)

本目录的三份模板共用交付的 Qwen3.8-27B W8 权重。官方 Qwen3.5-27B 使用另一组模型配置和量化范围，入口见[官方 27B](../../README.md#官方-qwen35-27b)。

## 配置说明

<a id="配置与权重"></a>

三份模板使用同一份 W8 权重，没有 FP16 运行模板。比较以 `w8a16_attn_mlp.toml` 为起点：MLP 的 gate/up/down 和 full-attention 的 Q/K/V/O 投影为 INT8 + FP16 scale；**全部 GDN 投影、embedding、输出头、norm、conv 保留 FP16**，激活为 FP16。官方 Qwen3.5-27B 的统一 W8 还量化 GDN，使用不同模型配置和权重。

| 配置 | 相对基础模板的差异 | 权重获取步骤 |
|---|---|---|
| [w8a16_attn_mlp.toml](w8a16_attn_mlp.toml) | 基准：自然 P、D4、默认根核数 8 | [配套 W8 包，或从配套浮点父转换](#weights) |
| [w8a16_attn_mlp_tp4.toml](w8a16_attn_mlp_tp4.toml) | 更换 TP4 profile，`num_cores=4`；D4 改为 D31 | [同一份 W8 权重](#weights) |
| [w8a16_attn_mlp_tp6.toml](w8a16_attn_mlp_tp6.toml) | 更换 TP6 profile，`num_cores=6`；D4 改为 D31 | [同一份 W8 权重](#weights) |

P 是输入 token 数，D 是 prefill 后的额外 decode 次数，固定 D 时输出 D+1 个 token；W/R 是预热和测量次数。三份模板均自动分块、W2/R3、EOS 停止、ACC16。

## 准备与运行

<a id="weights"></a>

<a id="准备权重并运行"></a>

配套包放入 `$RPU_MODEL_CACHE/Qwen3.8-27B-w8a16`。需要从浮点父转换时，使用交付的 BF16 legacy27 父及其完整 config、tokenizer；公开来源尚未登记。

在仓库根目录执行，`DST` 为新目录；已有配套 W8 包时跳过转换：

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache
SRC=/path/to/original-exact-legacy27-bf16
DST="$RPU_MODEL_CACHE/Qwen3.8-27B-w8a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --profile w8_main_fp16_gdn_legacy
python3 examples/run.py --config examples/configs/qwen3_5/legacy/27b/w8a16_attn_mlp.toml --check-config
python3 examples/run.py --config examples/configs/qwen3_5/legacy/27b/w8a16_attn_mlp.toml
```

转换器检查 legacy27 的模型形状、层类型和 RoPE，写入 W8 主干/FP16 GDN 声明；源目录应为未量化权重。转换可在 CPU 执行，推理在 设备板卡执行。

### 关键配置项

通用语义见[参数说明](../../../PARAMETERS.md)，这里列出本模型的值与关联。

| 层次 / 字段 | 模板值与有效省略值 | 范围与关联 |
|---|---|---|
| 配方：`example.registry_alias/dtype/profile_id` | alias 为 `qwen3_8-27b-w8a16`，dtype 为 `w8a16`；三项必填 | profile 按 TP8/4/6 选择；量化范围是 W8 主干、FP16 GDN，不使用官方 27B alias |
| 输入：`input.prompt/prefill_tokens` | 模板用短 prompt，未固定 P；省略 P 时保留自然长度 | P 为正整数；指定后按提示词 token 重复或截取 |
| 生成：`input.decode_steps/stop_on_eos` | 基础 D4，TP4/6 为 D31；EOS 省略为 true；D 省略为 4 | D≥0；核数比较时统一 D；固定输出数时显式关闭 EOS |
| 拓扑：`rpu_execution.model.num_cores` | 基础省略为 8，变体显式 4/6 | 与 profile 配对；TP6 的 MLP6，Attention/GDN/head4 |
| 累加：`rpu_execution.prefill.linear_acc32` | 模板省略为 false，即 ACC16 | true 为 ACC32；改变文本 Linear 累加，权重目录不变 |
| 分块：`rpu_execution.prefill.chunk_size` | 模板和省略值为 auto | 显式值为 16 的正整数倍，并受模型布局约束 |
| 测量：`cpu.threads`、`run.warmup/runs/inference_mode` | CPU 沿用启动环境；模板 W2/R3；省略 W/R 为 2/1、inference_mode 为 true | warmup≥0、runs≥1 |

可在配置副本修改 `input.prompt`、`input.decode_steps`、`run.output_dir`、`run.warmup/runs` 和 `cpu.threads`。核数与 `profile_id` 成对选择，改变安装参数后使用新进程。模板已包含量化入口的 opt-in。
