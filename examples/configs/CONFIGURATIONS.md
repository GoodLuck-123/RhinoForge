# 完整配置索引

[模型指南](README.md) · [权重准备](WEIGHTS.md) · [通用参数](PARAMETERS.md) · [命名规则](NAMING.md)

共 280 份 TOML：279 份运行模板，1 份资产元数据。运行前复制模板并填写权重与输入路径；模型指南列出量化范围、输入约束和可选参数。

## dinov3

[配置与权重说明](dinov3/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [dinov3/vit_b/fp16.toml](dinov3/vit_b/fp16.toml) | `fp16` | 见模板 | 运行模板 |

## g05

[配置与权重说明](g05/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [g05/2cam/fp16.toml](g05/2cam/fp16.toml) | `fp16` | 相机=2；T 列表=[32, 52, 64, 128, 192, 256] | 运行模板 |
| [g05/2cam/w8a16.toml](g05/2cam/w8a16.toml) | `w8a16` | 相机=2；T 列表=[32, 52, 64, 128, 192, 256] | 运行模板 |
| [g05/2cam/w8a16_action_nvfp4.toml](g05/2cam/w8a16_action_nvfp4.toml) | `w8a16_nvfp4` | 相机=2；T 列表=[32, 52, 64, 128, 192, 256] | 运行模板 |
| [g05/3cam/fp16.toml](g05/3cam/fp16.toml) | `fp16` | 相机=3；T 列表=[32, 52, 64, 128, 192, 256] | 运行模板 |
| [g05/3cam/w8a16.toml](g05/3cam/w8a16.toml) | `w8a16` | 相机=3；T 列表=[32, 52, 64, 128, 192, 256] | 运行模板 |
| [g05/3cam/w8a16_action_nvfp4.toml](g05/3cam/w8a16_action_nvfp4.toml) | `w8a16_nvfp4` | 相机=3；T 列表=[32, 52, 64, 128, 192, 256] | 运行模板 |
| [g05/base/fp16.toml](g05/base/fp16.toml) | `—` | 见模板 | 资产元数据 |
| [g05/legacy/fp16.toml](g05/legacy/fp16.toml) | `fp16` | 见模板 | 运行模板 |

## gemma4

[配置与权重说明](gemma4/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [gemma4/e4b/fp16.toml](gemma4/e4b/fp16.toml) | `fp16` | D=4 | 运行模板 |

## gr00t

[配置与权重说明](gr00t/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [gr00t/n1_7_droid/224/fp16.toml](gr00t/n1_7_droid/224/fp16.toml) | `fp16` | 相机=3；图像=224；T=32 | 运行模板 |
| [gr00t/n1_7_droid/224/w8a16.toml](gr00t/n1_7_droid/224/w8a16.toml) | `w8a16` | 相机=3；图像=224；T=32 | 运行模板 |
| [gr00t/n1_7_droid/256/fp16.toml](gr00t/n1_7_droid/256/fp16.toml) | `fp16` | 相机=3；图像=256；T=32 | 运行模板 |
| [gr00t/n1_7_droid/256/w8a16.toml](gr00t/n1_7_droid/256/w8a16.toml) | `w8a16` | 相机=3；图像=256；T=32 | 运行模板 |

## halo

[配置与权重说明](halo/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [halo/action_expert/fp16.toml](halo/action_expert/fp16.toml) | `fp16` | 见模板 | 运行模板 |

## hy_embodied

[配置与权重说明](hy_embodied/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [hy_embodied/umi/fp16.toml](hy_embodied/umi/fp16.toml) | `fp16` | 见模板 | 运行模板 |

## hy_vla

[配置与权重说明](hy_vla/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [hy_vla/2cam/fp16.toml](hy_vla/2cam/fp16.toml) | `fp16` | 相机=2；T=32 | 运行模板 |
| [hy_vla/2cam/w8a16.toml](hy_vla/2cam/w8a16.toml) | `w8a16` | 相机=2；T=32 | 运行模板 |
| [hy_vla/2cam/w8a16_action_nvfp4.toml](hy_vla/2cam/w8a16_action_nvfp4.toml) | `w8a16_nvfp4` | 相机=2；T=32 | 运行模板 |
| [hy_vla/3cam/fp16.toml](hy_vla/3cam/fp16.toml) | `fp16` | 相机=3；T=32 | 运行模板 |
| [hy_vla/3cam/w8a16.toml](hy_vla/3cam/w8a16.toml) | `w8a16` | 相机=3；T=32 | 运行模板 |
| [hy_vla/3cam/w8a16_action_nvfp4.toml](hy_vla/3cam/w8a16_action_nvfp4.toml) | `w8a16_nvfp4` | 相机=3；T=32 | 运行模板 |

## internvla

[配置与权重说明](internvla/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [internvla/navdp/w8a16_system2.toml](internvla/navdp/w8a16_system2.toml) | System2 W8；其余见模型页 | 见模板 | 运行模板 |

## lingbot

[配置与权重说明](lingbot/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [lingbot/legacy/exact_z2/6b/w8a16.toml](lingbot/legacy/exact_z2/6b/w8a16.toml) | `w8a16` | 见模板 | 运行模板 |
| [lingbot/v1/4b/fp16_cpu_vision.toml](lingbot/v1/4b/fp16_cpu_vision.toml) | `cpu-fp32+rpu-fp16` | 见模板 | 运行模板 |
| [lingbot/v2/2cam_t128/fp16.toml](lingbot/v2/2cam_t128/fp16.toml) | `fp16` | 相机=2；有效 T=128 | 运行模板 |
| [lingbot/v2/2cam_t128/w8a16.toml](lingbot/v2/2cam_t128/w8a16.toml) | `w8a16` | 相机=2；有效 T=128 | 运行模板 |
| [lingbot/v2/2cam_t16/fp16.toml](lingbot/v2/2cam_t16/fp16.toml) | `fp16` | 相机=2；有效 T=16 | 运行模板 |
| [lingbot/v2/2cam_t16/w8a16.toml](lingbot/v2/2cam_t16/w8a16.toml) | `w8a16` | 相机=2；有效 T=16 | 运行模板 |
| [lingbot/v2/2cam_t160/fp16.toml](lingbot/v2/2cam_t160/fp16.toml) | `fp16` | 相机=2；有效 T=160 | 运行模板 |
| [lingbot/v2/2cam_t160/w8a16.toml](lingbot/v2/2cam_t160/w8a16.toml) | `w8a16` | 相机=2；有效 T=160 | 运行模板 |
| [lingbot/v2/2cam_t32/fp16.toml](lingbot/v2/2cam_t32/fp16.toml) | `fp16` | 相机=2；有效 T=32 | 运行模板 |
| [lingbot/v2/2cam_t32/w8a16.toml](lingbot/v2/2cam_t32/w8a16.toml) | `w8a16` | 相机=2；有效 T=32 | 运行模板 |
| [lingbot/v2/2cam_t64/fp16.toml](lingbot/v2/2cam_t64/fp16.toml) | `fp16` | 相机=2；有效 T=64 | 运行模板 |
| [lingbot/v2/2cam_t64/w8a16.toml](lingbot/v2/2cam_t64/w8a16.toml) | `w8a16` | 相机=2；有效 T=64 | 运行模板 |
| [lingbot/v2/3cam_t128/fp16.toml](lingbot/v2/3cam_t128/fp16.toml) | `fp16` | 相机=3；有效 T=128 | 运行模板 |
| [lingbot/v2/3cam_t128/w8a16.toml](lingbot/v2/3cam_t128/w8a16.toml) | `w8a16` | 相机=3；有效 T=128 | 运行模板 |
| [lingbot/v2/3cam_t16/fp16.toml](lingbot/v2/3cam_t16/fp16.toml) | `fp16` | 相机=3；有效 T=16 | 运行模板 |
| [lingbot/v2/3cam_t16/w8a16.toml](lingbot/v2/3cam_t16/w8a16.toml) | `w8a16` | 相机=3；有效 T=16 | 运行模板 |
| [lingbot/v2/3cam_t160/fp16.toml](lingbot/v2/3cam_t160/fp16.toml) | `fp16` | 相机=3；有效 T=160 | 运行模板 |
| [lingbot/v2/3cam_t160/w8a16.toml](lingbot/v2/3cam_t160/w8a16.toml) | `w8a16` | 相机=3；有效 T=160 | 运行模板 |
| [lingbot/v2/3cam_t3/fp16.toml](lingbot/v2/3cam_t3/fp16.toml) | `fp16` | 相机=3；有效 T=3 | 运行模板 |
| [lingbot/v2/3cam_t3/w8a16.toml](lingbot/v2/3cam_t3/w8a16.toml) | `w8a16` | 相机=3；有效 T=3 | 运行模板 |
| [lingbot/v2/3cam_t32/fp16.toml](lingbot/v2/3cam_t32/fp16.toml) | `fp16` | 相机=3；有效 T=32 | 运行模板 |
| [lingbot/v2/3cam_t32/w8a16.toml](lingbot/v2/3cam_t32/w8a16.toml) | `w8a16` | 相机=3；有效 T=32 | 运行模板 |
| [lingbot/v2/3cam_t64/fp16.toml](lingbot/v2/3cam_t64/fp16.toml) | `fp16` | 相机=3；有效 T=64 | 运行模板 |
| [lingbot/v2/3cam_t64/w8a16.toml](lingbot/v2/3cam_t64/w8a16.toml) | `w8a16` | 相机=3；有效 T=64 | 运行模板 |

## llama

[配置与权重说明](llama/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [llama/3_2_1b/fp16.toml](llama/3_2_1b/fp16.toml) | `fp16` | D=4 | 运行模板 |

## pi05

[配置与权重说明](pi05/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [pi05/2cam/fp16.toml](pi05/2cam/fp16.toml) | `fp16` | 相机=2；T=32 | 运行模板 |
| [pi05/2cam/w8a16.toml](pi05/2cam/w8a16.toml) | `w8a16` | 相机=2；T=32 | 运行模板 |
| [pi05/2cam/w8a16_action_nvfp4.toml](pi05/2cam/w8a16_action_nvfp4.toml) | `w8a16_action_nvfp4` | 相机=2；T=32 | 运行模板 |
| [pi05/2cam/w8a16_prefill_w8a8_action_nvfp4.toml](pi05/2cam/w8a16_prefill_w8a8_action_nvfp4.toml) | `w8a16_prefill_w8a8_action_nvfp4` | 相机=2；T=32 | 运行模板 |
| [pi05/3cam/fp16.toml](pi05/3cam/fp16.toml) | `fp16` | 相机=3；T=32 | 运行模板 |
| [pi05/3cam/w8a16.toml](pi05/3cam/w8a16.toml) | `w8a16` | 相机=3；T=32 | 运行模板 |
| [pi05/3cam/w8a16_action_nvfp4.toml](pi05/3cam/w8a16_action_nvfp4.toml) | `w8a16_action_nvfp4` | 相机=3；T=32 | 运行模板 |
| [pi05/3cam/w8a16_prefill_w8a8_action_nvfp4.toml](pi05/3cam/w8a16_prefill_w8a8_action_nvfp4.toml) | `w8a16_prefill_w8a8_action_nvfp4` | 相机=3；T=32 | 运行模板 |
| [pi05/generic/3cam/fp16_tp4.toml](pi05/generic/3cam/fp16_tp4.toml) | `fp16` | 相机=3；T=32 | 运行模板 |
| [pi05/generic/3cam/fp16_tp6.toml](pi05/generic/3cam/fp16_tp6.toml) | `fp16` | 相机=3；T=32 | 运行模板 |
| [pi05/generic/3cam/w8a16_tp4.toml](pi05/generic/3cam/w8a16_tp4.toml) | `w8a16` | 相机=3；T=32 | 运行模板 |
| [pi05/generic/3cam/w8a16_tp6.toml](pi05/generic/3cam/w8a16_tp6.toml) | `w8a16` | 相机=3；T=32 | 运行模板 |

## qwen3

[配置与权重说明](qwen3/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [qwen3/text/0_6b/fp16.toml](qwen3/text/0_6b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3/text/0_6b/fp16_instruct.toml](qwen3/text/0_6b/fp16_instruct.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3/text/0_6b/w4a16_text_head_rtn_g32.toml](qwen3/text/0_6b/w4a16_text_head_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/0_6b/w4a16_text_rtn_g32.toml](qwen3/text/0_6b/w4a16_text_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/0_6b/w8a16_text_head.toml](qwen3/text/0_6b/w8a16_text_head.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3/text/14b/w4a16_text_head_rtn_g32.toml](qwen3/text/14b/w4a16_text_head_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/14b/w4a16_text_rtn_g32.toml](qwen3/text/14b/w4a16_text_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/14b/w8a16_text_head.toml](qwen3/text/14b/w8a16_text_head.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3/text/1_7b/fp16.toml](qwen3/text/1_7b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3/text/1_7b/w4a16_text_head_rtn_g32.toml](qwen3/text/1_7b/w4a16_text_head_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/1_7b/w4a16_text_rtn_g32.toml](qwen3/text/1_7b/w4a16_text_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/1_7b/w8a16_text_head.toml](qwen3/text/1_7b/w8a16_text_head.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3/text/32b/w4a16_text_head_rtn_g32.toml](qwen3/text/32b/w4a16_text_head_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/32b/w4a16_text_rtn_g32.toml](qwen3/text/32b/w4a16_text_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/32b/w8a16_text_head.toml](qwen3/text/32b/w8a16_text_head.toml) | `w8a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/4b/fp16.toml](qwen3/text/4b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3/text/4b/fp16_instruct.toml](qwen3/text/4b/fp16_instruct.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3/text/4b/w4a16_text_head_rtn_g32.toml](qwen3/text/4b/w4a16_text_head_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/4b/w4a16_text_rtn_g32.toml](qwen3/text/4b/w4a16_text_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/4b/w8a16_text_head.toml](qwen3/text/4b/w8a16_text_head.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3/text/8b/fp16.toml](qwen3/text/8b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3/text/8b/w4a16_text_head_rtn_g32.toml](qwen3/text/8b/w4a16_text_head_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/8b/w4a16_text_rtn_g32.toml](qwen3/text/8b/w4a16_text_rtn_g32.toml) | `w4a16` | P=128；D=7 | 运行模板 |
| [qwen3/text/8b/w8a16_text_head.toml](qwen3/text/8b/w8a16_text_head.toml) | `w8a16` | D=4 | 运行模板 |

## qwen3_5

[配置与权重说明](qwen3_5/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [qwen3_5/legacy/27b/w8a16_attn_mlp.toml](qwen3_5/legacy/27b/w8a16_attn_mlp.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3_5/legacy/27b/w8a16_attn_mlp_tp4.toml](qwen3_5/legacy/27b/w8a16_attn_mlp_tp4.toml) | `w8a16` | D=31 | 运行模板 |
| [qwen3_5/legacy/27b/w8a16_attn_mlp_tp6.toml](qwen3_5/legacy/27b/w8a16_attn_mlp_tp6.toml) | `w8a16` | D=31 | 运行模板 |
| [qwen3_5/text/0_8b/fp16.toml](qwen3_5/text/0_8b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_5/text/0_8b/fp16_p4096_d4.toml](qwen3_5/text/0_8b/fp16_p4096_d4.toml) | `fp16` | P=4096；D=4 | 运行模板 |
| [qwen3_5/text/0_8b/fp16_p4096_d4096.toml](qwen3_5/text/0_8b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/0_8b/fp16_p64_d4096.toml](qwen3_5/text/0_8b/fp16_p64_d4096.toml) | `fp16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/0_8b/fp16_tp4.toml](qwen3_5/text/0_8b/fp16_tp4.toml) | `fp16` | D=32 | 运行模板 |
| [qwen3_5/text/0_8b/fp16_tp6.toml](qwen3_5/text/0_8b/fp16_tp6.toml) | `fp16` | D=32 | 运行模板 |
| [qwen3_5/text/0_8b/w4a16_text_rtn_g32.toml](qwen3_5/text/0_8b/w4a16_text_rtn_g32.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/0_8b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/0_8b/w4a16_text_rtn_g32_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/0_8b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/0_8b/w4a16_text_rtn_g32_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/0_8b/w8a16_text.toml](qwen3_5/text/0_8b/w8a16_text.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3_5/text/0_8b/w8a16_text_p4096_d4096.toml](qwen3_5/text/0_8b/w8a16_text_p4096_d4096.toml) | `w8a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/0_8b/w8a16_text_p64_d4096.toml](qwen3_5/text/0_8b/w8a16_text_p64_d4096.toml) | `w8a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/27b/w4a16_text_rtn_g32.toml](qwen3_5/text/27b/w4a16_text_rtn_g32.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/27b/w4a16_text_rtn_g32_acc32_text.toml](qwen3_5/text/27b/w4a16_text_rtn_g32_acc32_text.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/27b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/27b/w4a16_text_rtn_g32_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/27b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/27b/w4a16_text_rtn_g32_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text.toml](qwen3_5/text/27b/w8a16_text.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_acc32_text_p29_d31.toml](qwen3_5/text/27b/w8a16_text_acc32_text_p29_d31.toml) | `w8a16` | P=29；D=31 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_p29_d31.toml](qwen3_5/text/27b/w8a16_text_p29_d31.toml) | `w8a16` | P=29；D=31 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_p4096_d4.toml](qwen3_5/text/27b/w8a16_text_p4096_d4.toml) | `w8a16` | P=4096；D=4 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_p4096_d4096.toml](qwen3_5/text/27b/w8a16_text_p4096_d4096.toml) | `w8a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_p64_d4096.toml](qwen3_5/text/27b/w8a16_text_p64_d4096.toml) | `w8a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_tp4.toml](qwen3_5/text/27b/w8a16_text_tp4.toml) | `w8a16` | D=32 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_tp4_p29_d31.toml](qwen3_5/text/27b/w8a16_text_tp4_p29_d31.toml) | `w8a16` | P=29；D=31 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_tp6.toml](qwen3_5/text/27b/w8a16_text_tp6.toml) | `w8a16` | D=32 | 运行模板 |
| [qwen3_5/text/27b/w8a16_text_tp6_p29_d31.toml](qwen3_5/text/27b/w8a16_text_tp6_p29_d31.toml) | `w8a16` | P=29；D=31 | 运行模板 |
| [qwen3_5/text/2b/fp16.toml](qwen3_5/text/2b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_5/text/2b/fp16_chunk640_p4096_d4.toml](qwen3_5/text/2b/fp16_chunk640_p4096_d4.toml) | `fp16` | P=4096；D=4 | 运行模板 |
| [qwen3_5/text/2b/fp16_p4096_d4096.toml](qwen3_5/text/2b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/2b/fp16_p64_d4096.toml](qwen3_5/text/2b/fp16_p64_d4096.toml) | `fp16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/2b/fp16_tp4.toml](qwen3_5/text/2b/fp16_tp4.toml) | `fp16` | D=32 | 运行模板 |
| [qwen3_5/text/2b/fp16_tp6.toml](qwen3_5/text/2b/fp16_tp6.toml) | `fp16` | D=32 | 运行模板 |
| [qwen3_5/text/2b/w4a16_mlp_awq_g128.toml](qwen3_5/text/2b/w4a16_mlp_awq_g128.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/2b/w4a16_mlp_awq_g128_p4096_d4096.toml](qwen3_5/text/2b/w4a16_mlp_awq_g128_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/2b/w4a16_mlp_awq_g128_p64_d4096.toml](qwen3_5/text/2b/w4a16_mlp_awq_g128_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32_p4096_d4096.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32_p64_d4096.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/2b/w4a16_text_rtn_g32.toml](qwen3_5/text/2b/w4a16_text_rtn_g32.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/2b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/2b/w4a16_text_rtn_g32_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/2b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/2b/w4a16_text_rtn_g32_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/2b/w8a16_text.toml](qwen3_5/text/2b/w8a16_text.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3_5/text/2b/w8a16_text_p4096_d4096.toml](qwen3_5/text/2b/w8a16_text_p4096_d4096.toml) | `w8a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/2b/w8a16_text_p64_d4096.toml](qwen3_5/text/2b/w8a16_text_p64_d4096.toml) | `w8a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | `mixed-e4m3` | D=31 | 运行模板 |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml) | `mixed-e4m3` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p64_d4096.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p64_d4096.toml) | `mixed-e4m3` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_tp4.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_tp4.toml) | `mixed-e4m3` | D=31 | 运行模板 |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_tp6.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_tp6.toml) | `mixed-e4m3` | D=31 | 运行模板 |
| [qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml](qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml) | `w4a16` | D=31 | 运行模板 |
| [qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128_p4096_d4096.toml](qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128_p64_d4096.toml](qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml](qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml) | `w4a16` | D=31 | 运行模板 |
| [qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p4096_d4096.toml](qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p64_d4096.toml](qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/4b/fp16.toml](qwen3_5/text/4b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_5/text/4b/fp16_p4096_d4.toml](qwen3_5/text/4b/fp16_p4096_d4.toml) | `fp16` | P=4096；D=4 | 运行模板 |
| [qwen3_5/text/4b/fp16_p4096_d4096.toml](qwen3_5/text/4b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/4b/fp16_p64_d4096.toml](qwen3_5/text/4b/fp16_p64_d4096.toml) | `fp16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/4b/fp16_tp4.toml](qwen3_5/text/4b/fp16_tp4.toml) | `fp16` | D=32 | 运行模板 |
| [qwen3_5/text/4b/fp16_tp6.toml](qwen3_5/text/4b/fp16_tp6.toml) | `fp16` | D=32 | 运行模板 |
| [qwen3_5/text/4b/w4a16_mlp_awq_g128.toml](qwen3_5/text/4b/w4a16_mlp_awq_g128.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/4b/w4a16_mlp_awq_g128_p4096_d4096.toml](qwen3_5/text/4b/w4a16_mlp_awq_g128_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/4b/w4a16_mlp_awq_g128_p64_d4096.toml](qwen3_5/text/4b/w4a16_mlp_awq_g128_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/4b/w4a16_text_rtn_g32.toml](qwen3_5/text/4b/w4a16_text_rtn_g32.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/4b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/4b/w4a16_text_rtn_g32_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/4b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/4b/w4a16_text_rtn_g32_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/4b/w8a16_text.toml](qwen3_5/text/4b/w8a16_text.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3_5/text/4b/w8a16_text_p4096_d4096.toml](qwen3_5/text/4b/w8a16_text_p4096_d4096.toml) | `w8a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/4b/w8a16_text_p64_d4096.toml](qwen3_5/text/4b/w8a16_text_p64_d4096.toml) | `w8a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/9b/fp16.toml](qwen3_5/text/9b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_5/text/9b/fp16_p4096_d4.toml](qwen3_5/text/9b/fp16_p4096_d4.toml) | `fp16` | P=4096；D=4 | 运行模板 |
| [qwen3_5/text/9b/fp16_p4096_d4096.toml](qwen3_5/text/9b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/9b/fp16_p64_d4096.toml](qwen3_5/text/9b/fp16_p64_d4096.toml) | `fp16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/9b/fp16_tp4.toml](qwen3_5/text/9b/fp16_tp4.toml) | `fp16` | D=32 | 运行模板 |
| [qwen3_5/text/9b/fp16_tp6.toml](qwen3_5/text/9b/fp16_tp6.toml) | `fp16` | D=32 | 运行模板 |
| [qwen3_5/text/9b/w4a16_mlp_awq_g128.toml](qwen3_5/text/9b/w4a16_mlp_awq_g128.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/9b/w4a16_mlp_awq_g128_p4096_d4096.toml](qwen3_5/text/9b/w4a16_mlp_awq_g128_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/9b/w4a16_mlp_awq_g128_p64_d4096.toml](qwen3_5/text/9b/w4a16_mlp_awq_g128_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/9b/w4a16_text_rtn_g32.toml](qwen3_5/text/9b/w4a16_text_rtn_g32.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/text/9b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/9b/w4a16_text_rtn_g32_p4096_d4096.toml) | `w4a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/9b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/9b/w4a16_text_rtn_g32_p64_d4096.toml) | `w4a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/text/9b/w8a16_text.toml](qwen3_5/text/9b/w8a16_text.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3_5/text/9b/w8a16_text_p4096_d4096.toml](qwen3_5/text/9b/w8a16_text_p4096_d4096.toml) | `w8a16` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/text/9b/w8a16_text_p64_d4096.toml](qwen3_5/text/9b/w8a16_text_p64_d4096.toml) | `w8a16` | P=64；D=4096 | 运行模板 |
| [qwen3_5/vl/0_8b/fp16.toml](qwen3_5/vl/0_8b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_5/vl/0_8b/fp16_p4096_d4096.toml](qwen3_5/vl/0_8b/fp16_p4096_d4096.toml) | `fp16` | 图像=448；P=4096；D=4096 | 运行模板 |
| [qwen3_5/vl/0_8b/fp16_p512_d192.toml](qwen3_5/vl/0_8b/fp16_p512_d192.toml) | `fp16` | 图像=448；P=512；D=192 | 运行模板 |
| [qwen3_5/vl/0_8b/w4a16_text_rtn_g32.toml](qwen3_5/vl/0_8b/w4a16_text_rtn_g32.toml) | `w4a16` | D=32 | 运行模板 |
| [qwen3_5/vl/0_8b/w8a16_text.toml](qwen3_5/vl/0_8b/w8a16_text.toml) | `w8a16` | D=32 | 运行模板 |
| [qwen3_5/vl/27b/w4a16_text_rtn_g32.toml](qwen3_5/vl/27b/w4a16_text_rtn_g32.toml) | `w4a16` | D=32 | 运行模板 |
| [qwen3_5/vl/27b/w4a16_text_rtn_g32_acc32_text_vision.toml](qwen3_5/vl/27b/w4a16_text_rtn_g32_acc32_text_vision.toml) | `w4a16` | D=32 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text.toml](qwen3_5/vl/27b/w8a16_text.toml) | `w8a16` | D=32 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text_acc32_text.toml](qwen3_5/vl/27b/w8a16_text_acc32_text.toml) | `w8a16` | D=32 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text_acc32_vision.toml](qwen3_5/vl/27b/w8a16_text_acc32_vision.toml) | `w8a16` | D=32 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text_d4.toml](qwen3_5/vl/27b/w8a16_text_d4.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text_image_mascot_d31.toml](qwen3_5/vl/27b/w8a16_text_image_mascot_d31.toml) | `w8a16` | D=31 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text_p4096_d4096.toml](qwen3_5/vl/27b/w8a16_text_p4096_d4096.toml) | `w8a16` | 图像=448；P=4096；D=4096 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text_p512_d192.toml](qwen3_5/vl/27b/w8a16_text_p512_d192.toml) | `w8a16` | 图像=448；P=512；D=192 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text_tp4_image_mascot_d31.toml](qwen3_5/vl/27b/w8a16_text_tp4_image_mascot_d31.toml) | `w8a16` | D=31 | 运行模板 |
| [qwen3_5/vl/27b/w8a16_text_tp6_image_mascot_d31.toml](qwen3_5/vl/27b/w8a16_text_tp6_image_mascot_d31.toml) | `w8a16` | D=31 | 运行模板 |
| [qwen3_5/vl/2b/fp16.toml](qwen3_5/vl/2b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_5/vl/2b/fp16_p4096_d4096.toml](qwen3_5/vl/2b/fp16_p4096_d4096.toml) | `fp16` | 图像=448；P=4096；D=4096 | 运行模板 |
| [qwen3_5/vl/2b/fp16_p512_d192.toml](qwen3_5/vl/2b/fp16_p512_d192.toml) | `fp16` | 图像=448；P=512；D=192 | 运行模板 |
| [qwen3_5/vl/2b/w4a16_text_rtn_g32.toml](qwen3_5/vl/2b/w4a16_text_rtn_g32.toml) | `w4a16` | D=32 | 运行模板 |
| [qwen3_5/vl/2b/w8a16_text.toml](qwen3_5/vl/2b/w8a16_text.toml) | `w8a16` | D=32 | 运行模板 |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | `mixed-e4m3` | D=4 | 运行模板 |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml) | `mixed-e4m3` | P=4096；D=4096 | 运行模板 |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_tp4.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_tp4.toml) | `mixed-e4m3` | D=4 | 运行模板 |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_tp6.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_tp6.toml) | `mixed-e4m3` | D=4 | 运行模板 |
| [qwen3_5/vl/35b_a3b/w4a16_experts_gptq_g128.toml](qwen3_5/vl/35b_a3b/w4a16_experts_gptq_g128.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/vl/35b_a3b/w4a16_experts_rtn_g128.toml](qwen3_5/vl/35b_a3b/w4a16_experts_rtn_g128.toml) | `w4a16` | D=4 | 运行模板 |
| [qwen3_5/vl/4b/fp16.toml](qwen3_5/vl/4b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_5/vl/4b/fp16_p4096_d4096.toml](qwen3_5/vl/4b/fp16_p4096_d4096.toml) | `fp16` | 图像=448；P=4096；D=4096 | 运行模板 |
| [qwen3_5/vl/4b/fp16_p512_d192.toml](qwen3_5/vl/4b/fp16_p512_d192.toml) | `fp16` | 图像=448；P=512；D=192 | 运行模板 |
| [qwen3_5/vl/4b/w4a16_text_rtn_g32.toml](qwen3_5/vl/4b/w4a16_text_rtn_g32.toml) | `w4a16` | D=32 | 运行模板 |
| [qwen3_5/vl/4b/w8a16_text.toml](qwen3_5/vl/4b/w8a16_text.toml) | `w8a16` | D=32 | 运行模板 |
| [qwen3_5/vl/9b/fp16.toml](qwen3_5/vl/9b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_5/vl/9b/fp16_p4096_d4096.toml](qwen3_5/vl/9b/fp16_p4096_d4096.toml) | `fp16` | 图像=448；P=4096；D=4096 | 运行模板 |
| [qwen3_5/vl/9b/fp16_p512_d192.toml](qwen3_5/vl/9b/fp16_p512_d192.toml) | `fp16` | 图像=448；P=512；D=192 | 运行模板 |
| [qwen3_5/vl/9b/w8a16_text.toml](qwen3_5/vl/9b/w8a16_text.toml) | `w8a16` | D=32 | 运行模板 |

## qwen3_vl

[配置与权重说明](qwen3_vl/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [qwen3_vl/text/2b/fp16.toml](qwen3_vl/text/2b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/text/2b/fp16_p192_d192.toml](qwen3_vl/text/2b/fp16_p192_d192.toml) | `fp16` | P=192；D=192 | 运行模板 |
| [qwen3_vl/text/2b/fp16_p4096_d4096.toml](qwen3_vl/text/2b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_vl/text/2b/w4a16_text_awq_g32.toml](qwen3_vl/text/2b/w4a16_text_awq_g32.toml) | `w4a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | `w4a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/2b/w8a16_text.toml](qwen3_vl/text/2b/w8a16_text.toml) | `w8a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/2b/w8a16_text_head_vision.toml](qwen3_vl/text/2b/w8a16_text_head_vision.toml) | `w8a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | `w4a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/32b/w8a16_text_head_vision.toml](qwen3_vl/text/32b/w8a16_text_head_vision.toml) | `w8a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/4b/fp16.toml](qwen3_vl/text/4b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/text/4b/fp16_p192_d192.toml](qwen3_vl/text/4b/fp16_p192_d192.toml) | `fp16` | P=192；D=192 | 运行模板 |
| [qwen3_vl/text/4b/fp16_p4096_d4096.toml](qwen3_vl/text/4b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_vl/text/4b/w4a16_text_awq_g32.toml](qwen3_vl/text/4b/w4a16_text_awq_g32.toml) | `w4a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | `w4a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/4b/w8a16_text_head_vision.toml](qwen3_vl/text/4b/w8a16_text_head_vision.toml) | `w8a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/8b/fp16.toml](qwen3_vl/text/8b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/text/8b/fp16_p192_d192.toml](qwen3_vl/text/8b/fp16_p192_d192.toml) | `fp16` | P=192；D=192 | 运行模板 |
| [qwen3_vl/text/8b/fp16_p4096_d4096.toml](qwen3_vl/text/8b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_vl/text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | `w4a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/text/8b/w8a16_text_head_vision.toml](qwen3_vl/text/8b/w8a16_text_head_vision.toml) | `w8a16` | P=512；D=31 | 运行模板 |
| [qwen3_vl/vision/2b/fp16.toml](qwen3_vl/vision/2b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/vision/4b/fp16.toml](qwen3_vl/vision/4b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/vision/8b/fp16.toml](qwen3_vl/vision/8b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/vl/2b/fp16.toml](qwen3_vl/vl/2b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/vl/2b/fp16_p256_d192.toml](qwen3_vl/vl/2b/fp16_p256_d192.toml) | `fp16` | P=256；D=192 | 运行模板 |
| [qwen3_vl/vl/2b/fp16_p4096_d4096.toml](qwen3_vl/vl/2b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_vl/vl/2b/w4a16_text_awq_g32.toml](qwen3_vl/vl/2b/w4a16_text_awq_g32.toml) | `w4a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | `w4a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/2b/w8a16_text.toml](qwen3_vl/vl/2b/w8a16_text.toml) | `w8a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/2b/w8a16_text_head_vision.toml](qwen3_vl/vl/2b/w8a16_text_head_vision.toml) | `w8a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | `w4a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/32b/w8a16_text.toml](qwen3_vl/vl/32b/w8a16_text.toml) | `w8a16` | D=4 | 运行模板 |
| [qwen3_vl/vl/32b/w8a16_text_head_vision.toml](qwen3_vl/vl/32b/w8a16_text_head_vision.toml) | `w8a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/4b/fp16.toml](qwen3_vl/vl/4b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/vl/4b/fp16_p256_d192.toml](qwen3_vl/vl/4b/fp16_p256_d192.toml) | `fp16` | P=256；D=192 | 运行模板 |
| [qwen3_vl/vl/4b/fp16_p4096_d4096.toml](qwen3_vl/vl/4b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_vl/vl/4b/w4a16_text_awq_g32.toml](qwen3_vl/vl/4b/w4a16_text_awq_g32.toml) | `w4a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | `w4a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/4b/w8a16_text_head_vision.toml](qwen3_vl/vl/4b/w8a16_text_head_vision.toml) | `w8a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/8b/fp16.toml](qwen3_vl/vl/8b/fp16.toml) | `fp16` | D=4 | 运行模板 |
| [qwen3_vl/vl/8b/fp16_p256_d192.toml](qwen3_vl/vl/8b/fp16_p256_d192.toml) | `fp16` | P=256；D=192 | 运行模板 |
| [qwen3_vl/vl/8b/fp16_p4096_d4096.toml](qwen3_vl/vl/8b/fp16_p4096_d4096.toml) | `fp16` | P=4096；D=4096 | 运行模板 |
| [qwen3_vl/vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | `w4a16` | P=430；D=31 | 运行模板 |
| [qwen3_vl/vl/8b/w8a16_text_head_vision.toml](qwen3_vl/vl/8b/w8a16_text_head_vision.toml) | `w8a16` | P=430；D=31 | 运行模板 |

## rhinovla

[配置与权重说明](rhinovla/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [rhinovla/v3/controlled/fp16.toml](rhinovla/v3/controlled/fp16.toml) | `fp16` | FP16 | 独立运行配置 |
| [rhinovla/v3/controlled/w8a16_action.toml](rhinovla/v3/controlled/w8a16_action.toml) | `w8a16` | Action W8 | 独立运行配置 |
| [rhinovla/v3/controlled/w8a16_pipeline.toml](rhinovla/v3/controlled/w8a16_pipeline.toml) | `w8a16` | Pipeline W8 | 独立运行配置 |
| [rhinovla/v3/controlled/w8a16_pipeline_input_cache.toml](rhinovla/v3/controlled/w8a16_pipeline_input_cache.toml) | `w8a16` | Pipeline W8 with image input cache | 独立运行配置 |
| [rhinovla/v3/controlled/w8a16_pipeline_high.toml](rhinovla/v3/controlled/w8a16_pipeline_high.toml) | `w8a16` | Pipeline W8 with HIGH nonlinear operations | 独立运行配置 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused.toml) | `w8a16` | Pipeline W8 with HIGH fusion | 独立运行配置 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_k.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_k.toml) | `w8a16` | Pipeline W8 with vector K norm | 独立运行配置 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk.toml) | `w8a16` | Pipeline W8 with vector Q/K norm | 独立运行配置 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml) | `w8a16` | Pipeline W8 with partial RoPE | 独立运行配置 |
| [rhinovla/v1/fp16.toml](rhinovla/v1/fp16.toml) | `fp16` | 见模板 | 运行模板 |
| [rhinovla/v1/official_fp16.toml](rhinovla/v1/official_fp16.toml) | `fp16` | 见模板 | 运行模板 |
| [rhinovla/v3/fp16.toml](rhinovla/v3/fp16.toml) | `fp16` | 见模板 | 运行模板 |
| [rhinovla/v3/w8a16_action.toml](rhinovla/v3/w8a16_action.toml) | `w8a16` | 见模板 | 运行模板 |
| [rhinovla/v3/w8a16_action_adarms.toml](rhinovla/v3/w8a16_action_adarms.toml) | `w8a16` | 见模板 | 运行模板 |
| [rhinovla/v3/w8a16_pipeline.toml](rhinovla/v3/w8a16_pipeline.toml) | `w8a16` | 见模板 | 运行模板 |
| [rhinovla/v3/w8a16_pipeline_high_fused_vector_qk.toml](rhinovla/v3/w8a16_pipeline_high_fused_vector_qk.toml) | `w8a16` | 见模板 | 运行模板 |

## wall_oss

[配置与权重说明](wall_oss/README.md)

| 配置 | 配方 | 输入默认 | 用途 |
|---|---|---|---|
| [wall_oss/base/2cam/224/fp16.toml](wall_oss/base/2cam/224/fp16.toml) | `fp16` | 相机=2；图像=[224, 224]；T=32 | 运行模板 |
| [wall_oss/base/2cam/224/w8a16.toml](wall_oss/base/2cam/224/w8a16.toml) | `w8a16` | 相机=2；图像=[224, 224]；T=32 | 运行模板 |
| [wall_oss/base/2cam/224/w8a16_action_nvfp4.toml](wall_oss/base/2cam/224/w8a16_action_nvfp4.toml) | `w8a16_action_nvfp4` | 相机=2；图像=[224, 224]；T=32 | 运行模板 |
| [wall_oss/base/2cam/256/fp16.toml](wall_oss/base/2cam/256/fp16.toml) | `fp16` | 相机=2；图像=[256, 256]；T=32 | 运行模板 |
| [wall_oss/base/2cam/256/w8a16.toml](wall_oss/base/2cam/256/w8a16.toml) | `w8a16` | 相机=2；图像=[256, 256]；T=32 | 运行模板 |
| [wall_oss/base/2cam/256/w8a16_action_nvfp4.toml](wall_oss/base/2cam/256/w8a16_action_nvfp4.toml) | `w8a16_action_nvfp4` | 相机=2；图像=[256, 256]；T=32 | 运行模板 |
| [wall_oss/base/3cam/224/fp16.toml](wall_oss/base/3cam/224/fp16.toml) | `fp16` | 相机=3；图像=[224, 224]；T=32 | 运行模板 |
| [wall_oss/base/3cam/224/w8a16.toml](wall_oss/base/3cam/224/w8a16.toml) | `w8a16` | 相机=3；图像=[224, 224]；T=32 | 运行模板 |
| [wall_oss/base/3cam/224/w8a16_action_nvfp4.toml](wall_oss/base/3cam/224/w8a16_action_nvfp4.toml) | `w8a16_action_nvfp4` | 相机=3；图像=[224, 224]；T=32 | 运行模板 |
| [wall_oss/base/3cam/256/fp16.toml](wall_oss/base/3cam/256/fp16.toml) | `fp16` | 相机=3；图像=[256, 256]；T=32 | 运行模板 |
| [wall_oss/base/3cam/256/w8a16.toml](wall_oss/base/3cam/256/w8a16.toml) | `w8a16` | 相机=3；图像=[256, 256]；T=32 | 运行模板 |
| [wall_oss/base/3cam/256/w8a16_action_nvfp4.toml](wall_oss/base/3cam/256/w8a16_action_nvfp4.toml) | `w8a16_action_nvfp4` | 相机=3；图像=[256, 256]；T=32 | 运行模板 |
| [wall_oss/legacy/fp16.toml](wall_oss/legacy/fp16.toml) | `fp16` | 见模板 | 运行模板 |
| [wall_oss/legacy/w4a16_channel.toml](wall_oss/legacy/w4a16_channel.toml) | `w4a16` | 见模板 | 运行模板 |
| [wall_oss/legacy/w4a16_g32.toml](wall_oss/legacy/w4a16_g32.toml) | `w4a16-pgrp` | 见模板 | 运行模板 |
| [wall_oss/legacy/w8a16.toml](wall_oss/legacy/w8a16.toml) | `w8a16` | 见模板 | 运行模板 |
| [wall_oss/rtc/fp16.toml](wall_oss/rtc/fp16.toml) | `fp16` | 见模板 | 运行模板 |
| [wall_oss/rtc/w8a16.toml](wall_oss/rtc/w8a16.toml) | `w8a16` | 见模板 | 运行模板 |
| [wall_oss/sorting/fp16.toml](wall_oss/sorting/fp16.toml) | `—` | 见模板 | 运行模板 |
| [wall_oss/sorting/w8a16.toml](wall_oss/sorting/w8a16.toml) | `—` | 见模板 | 运行模板 |
| [wall_oss/sorting/w8a16_action_w4a16_g32.toml](wall_oss/sorting/w8a16_action_w4a16_g32.toml) | `—` | 见模板 | 运行模板 |

## 旧路径兼容

[路径对照](config_names.json)列出规范化后的名称。旧路径可继续用于 `--config` 和迁移命令的 `--template`；复制、编辑模板请使用当前路径。已存在的自定义文件仍按原路径读取。

以下 6 个重复文件已删除，旧命令转到对应模板：

| 旧路径 | 当前配置 |
|---|---|
| `wall_oss/base/fp16.toml` | [wall_oss/legacy/fp16.toml](wall_oss/legacy/fp16.toml) |
| `wall_oss/base/w8a16.toml` | [wall_oss/legacy/w8a16.toml](wall_oss/legacy/w8a16.toml) |
| `wall_oss/base/w4a16.toml` | [wall_oss/legacy/w4a16_channel.toml](wall_oss/legacy/w4a16_channel.toml) |
| `wall_oss/base/w4a16_pgrp.toml` | [wall_oss/legacy/w4a16_g32.toml](wall_oss/legacy/w4a16_g32.toml) |
| `lingbot/v2/6b_w8a16.toml` | [lingbot/legacy/exact_z2/6b/w8a16.toml](lingbot/legacy/exact_z2/6b/w8a16.toml) |
| `rhinovla/v3/w8a16.toml` | [rhinovla/v3/w8a16_action.toml](rhinovla/v3/w8a16_action.toml) |
