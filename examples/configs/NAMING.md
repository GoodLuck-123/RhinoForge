# 配置命名

[模型目录](README.md) · [全部配置](CONFIGURATIONS.md) · [参数说明](PARAMETERS.md) · [路径对照](config_names.json)

## 目录与文件名

目录按模型族、版本或任务、模型大小、输入规格分组，省略该模型不需要的层级。例如 `qwen3/text/4b/`、`wall_oss/base/3cam/224/`。同一目录按模型及输入规格组织，尺寸和相机规格不在每个文件名里重复；比较时还需对齐输入和计时条件。

文件名使用小写字母、数字和下划线，顺序为：

```text
<精度与模块范围>[_量化方法][_g分组大小][_执行选项][_输入规格].toml
```

只写区分配置所需的部分。`fp16.toml` 是浮点基准；存在多个量化范围时写明 `text`、`mlp`、`experts`、`action` 或 `pipeline`。混合精度组合按组件写明，不用笼统的 `mixed` 代替已有明确含义。

| 名称片段 | 含义 |
|---|---|
| `w8a16` / `w4a16` | 所选投影的 INT8 / INT4 权重，FP16 激活 |
| `text` / `head` / `vision` | 文本投影 / 输出头 / 视觉投影；具体层范围见模型页 |
| `mlp` / `experts` | MLP 投影 / MoE 路由专家，不等于全部文本模块 |
| `action_nvfp4` | 动作组件中指定投影使用 NVFP4，范围见该配方 |
| `fp8_e4m3` | 指定投影使用 FP8 E4M3，对应权重页的 mixed E4M3 配方 |
| `attn` / `gdn` | Attention / Gated DeltaNet；具体投影范围见模型页 |
| `rtn` / `awq` / `gptq` | 量化方法或所读取量化包的方法 |
| `g32` / `g128` | 量化分组大小；`channel` 表示逐输出通道 |
| `tp4` / `tp6` | 配置的设备根核数 |
| `acc32_text` / `acc32_vision` | 指定组件 Linear 使用 ACC32；二者都变时写 `acc32_text_vision` |
| `chunk640` | 显式执行分块大小 |
| `p64_d4096` | 固定 prefill 长度 P 和额外 decode 次数 D |
| `image_mascot` | 使用该配套图片和提示词；输入路径仍需填写 |
| `high_fused_vector_qk` | RhinoVLA 的高精度非线性、融合及 vector Q/K 归一化组合 |

执行后缀按 TP、ACC、chunk 排列，固定 P/D 放在末尾。同精度但不同图片、长度或计时条件的模板也要在模型页解释差异。

## 示例

| 配置 | 如何阅读 |
|---|---|
| `qwen3/text/4b/w4a16_text_rtn_g32.toml` | 4B 纯文本，文本主干 RTN W4/G32，输出头精度见配方 |
| `qwen3/text/4b/w4a16_text_head_rtn_g32.toml` | 同时量化文本主干与输出头 |
| `qwen3_5/text/2b/w4a16_mlp_awq_g128.toml` | 2B 纯文本，指定 MLP 范围使用 AWQ/G128 |
| `qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml` | MLP使用RTN W4/G32，Attention/GDN使用逐输出通道W8；G32只属于MLP |
| `qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml` | MoE 路由专家使用 GPTQ/G128 |
| `qwen3_5/vl/27b/w8a16_text_acc32_text.toml` | 文本投影使用 W8，文本 Linear 使用 ACC32；Vision 仍为 ACC16 |
| `qwen3_vl/vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml` | 文本主干 AWQ、输出头 RTN，均为 W4/G32；Vision W8 |
| `gr00t/n1_7_droid/256/fp16.toml` | N1.7 DROID、256×256 输入、FP16 |
| `internvla/navdp/w8a16_system2.toml` | NavDP 中 System2 的 decoder 使用 W8，其余组件精度见模型页 |

文件名用于选择，配置内容绑定实际模型与资产。权重目录、registry alias 和 profile ID 按加载接口保留自己的名称，不能从 TOML 文件名直接推导路径。

`generic/` 表示通用执行方式，例如 Pi0.5 TP4/6；`controlled/` 表示使用专用命令的独立配置；`legacy/` 表示保留独立资产、输入或调用约定的配置。

## 定位已有配置记录

[路径对照](config_names.json)按仓库根目录列出原路径到当前路径的对应关系。命令行、复制和编辑模板均使用当前路径。
