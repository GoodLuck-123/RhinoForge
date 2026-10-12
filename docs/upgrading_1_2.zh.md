# 升级到 v1.2.0

简体中文 | [English](upgrading_1_2.md)

当前源码版本为 v1.2.0。[发行说明](../CHANGELOG.md)列出主要变化；本页说明上一公开
版本的安装与迁移。发行包名 `rhinoforge`、导入名 `rpu_backend`、`torch.rpu` 和
现有运行时路径环境变量保持不变。

## 安装 v1.2.0

1. 按[入门指南](getting_started.zh.md)创建全新的 Python 3.12 环境，使用匹配的运行时
   资产构建。依赖以 [pyproject.toml](../pyproject.toml) 为准，包括 PyTorch 2.10.0 和
   Transformers 5.3.0；视觉/VLA/语音分别选择 `vision` / `vla` / `audio` 可选依赖。
2. 通过[资产申请入口](runtime_assets.zh.md#申请运行时资产)获取
   `RhinoForge-runtime-v1.2.0.tar.gz` 及校验和，核对完整性并保留 main、DDR-expansion
   两份资产及各自清单。Rhino Launch 仍为 1.0.0；准确源码和资产身份见
   [依赖清单](../release/runtime-dependencies.json)。
3. 更新 `RHINO_LAUNCH_ROOT`、`RPU_KERNEL_LIB_PATH` 和
   `RPU_KERNEL_LIB_PATH_EXPANSION`，重新构建扩展并启动新进程。
   旧环境与旧资产配套保留，供回退使用。
4. 运行 `python examples/verify_install.py --check-config`，再用
   `python examples/run.py --config PATH --check-config` 检查所选 TOML。
   在空闲设备上运行一次该配置，确认安装链路。

## 示例入口与路径

新命令使用 `python examples/run.py --config PATH`。原模型脚本和 `run_model.py`
保留为兼容入口；`run_model.py` 保留 `--list-targets` 和 Qwen3 默认配置。
GR00T、NavDP、独立 SigLIP 旧调用需按下文迁移后显式传入配置。
完整默认值与别名见 [legacy.py](../examples/common/legacy.py)。

以下重复文件已删除。运行命令的 `--config` 和迁移命令的 `--template` 仍可使用仓库内的旧路径，自动转到标准配置；复制或编辑文件时使用标准路径。已有自定义文件按原内容读取。表内路径均相对于 `examples/configs/`。

| 旧路径 | 标准路径 |
|---|---|
| `wall_oss/base/fp16.toml` | `wall_oss/legacy/fp16.toml` |
| `wall_oss/base/w8a16.toml` | `wall_oss/legacy/w8a16.toml` |
| `wall_oss/base/w4a16.toml` | `wall_oss/legacy/w4a16.toml` |
| `wall_oss/base/w4a16_pgrp.toml` | `wall_oss/legacy/w4a16_pgrp.toml` |
| `lingbot/v2/6b_w8a16.toml` | `lingbot/legacy/exact_z2/6b_w8a16.toml` |
| `rhinovla/v3/w8a16.toml` | `rhinovla/v3/expert_w8a16.toml` |

`rhinovla/v1/fp16.toml` 保留调用方提供 factory、checkpoint 与请求的原有含义。
[official_fp16.toml](../examples/configs/rhinovla/v1/official_fp16.toml) 使用
[v1 指南](../examples/configs/rhinovla/v1/README.md)指定的官方源码、checkpoint 和 fixture。

`g05/base/fp16.toml` 保留原 G05 源码集成方式：构建官方 CPU `G05PolicyQwen35`，调用
`patch_g05_policy_for_rpu` 后使用连续动作、离散动作与 CoT 请求入口。
当前 K1 动作示例使用另一套资产和输入，见 [G05 指南](../examples/configs/g05/README.md)。

## 需要迁移输入的旧配置

以下旧配置内容不能直接沿用，部分路径下已有新模板。按选定模板确认对应 checkpoint，
重新准备完整输入。

| 旧配置 | 迁移步骤 |
|---|---|
| `gr00t/n1_7_droid/fp16.toml` | 旧四图 DROID/Cosmos 与当前三图 Qwen 使用不同模型和 embodiment 输入。选定匹配 checkpoint 后准备相机与状态张量，将 `request.input_pt` 改为 `input.observation`，`request.seed/num_steps` 改为 `input.seed/num_steps`。见 [GR00T](../examples/configs/gr00t/README.md)。 |
| `internvla/navdp/fp16.toml` | 准备完整链路的 `input.fixture`、`input.asset_manifest`；将 `request.sample_num/seed` 保留为 `input.sample_num/seed`。旧 embeddings-only 组件文件不足以运行完整链路。见 [InternVLA](../examples/configs/internvla/README.md)。 |
| `siglip/pi05_component/fp16.toml` | 独立组件调用继续使用 SigLIP adapter；运行 Pi0.5 时，按 [Pi0.5 指南](../examples/configs/pi05/README.md)准备完整策略 batch。 |
| `lingbot/v2/6b_fp16.toml` | 先选相机数和语言长度，再提供 `input.checkpoint`、`input.training_config` 和匹配输入；可沿用旧图片/指令，并将 `proprioception` 转为 `state`，或显式提供新 `case` / `observation`。保持相机顺序、动作维度及归一化。见 [LingBot v2](../examples/configs/lingbot/v2/README.md)；exact-Z2 W8 单独使用。 |
| `qwen3_5/text/35b_a3b/mixed_e4m3_p4096_d4096.toml` | 将合成输入 `prompt_token_ids` / `prefill_fill_token_id` 改为 `prompt_file`、`prompt_sha256` 和 `input_ids_sha256`，设置 `run.timing_mode="forward_argmax"`。保留 checkpoint 及显式 prefill/decode 长度，记录新的文本输入。见 [Qwen3.5](../examples/configs/qwen3_5/README.md)。 |

## 迁移工具

[`examples/migrate_config.py`](../examples/migrate_config.py) 将旧 TOML 迁到显式选择的
当前模板。先按上表准备新的完整输入，再执行所需的一条命令；替换 `/path/to/...`
和旧配置文件名，输出父目录须已存在。

```bash
# GR00T：提供当前 checkpoint 对应的三图 observation。
python examples/migrate_config.py --source /path/to/old-gr00t.toml \
  --template examples/configs/gr00t/n1_7_droid/fp16.toml --output /path/to/new-gr00t.toml \
  --input observation=/path/to/three-camera-observation.pt

# NavDP：提供完整链路 fixture 与资产清单。
python examples/migrate_config.py --source /path/to/old-navdp.toml \
  --template examples/configs/internvla/navdp/fp16.toml --output /path/to/new-navdp.toml \
  --input fixture=/path/to/fullchain-fixture.pt --input asset_manifest=/path/to/assets.json

# SigLIP → Pi0.5：明确切换到完整策略并提供配套 batch 和 checkpoint。
python examples/migrate_config.py --source /path/to/old-siglip.toml \
  --template examples/configs/pi05/2cam/fp16.toml --output /path/to/new-pi05.toml \
  --input batch=/path/to/pi05-batch.pt --input checkpoint=/path/to/pi05-checkpoint

# LingBot：选择匹配相机数/语言长度的 FP16 模板，沿用旧 images/instruction/state。
python examples/migrate_config.py --source /path/to/old-lingbot.toml \
  --template examples/configs/lingbot/v2/3cam_t3/fp16.toml --output /path/to/new-lingbot.toml \
  --input checkpoint=/path/to/lingbot-checkpoint --input training_config=/path/to/training.yaml

# MoE：提供真实文本及其目标 token 序列的 SHA256。
python examples/migrate_config.py --source /path/to/old-moe.toml \
  --template examples/configs/qwen3_5/text/35b_a3b/mixed_e4m3_p4096_d4096.toml \
  --output /path/to/new-moe.toml --input prompt_file=/path/to/prompt.txt \
  --input input_ids_sha256="$INPUT_IDS_SHA256"
```

工具保留适用的 `run`、`cpu`、`logging`、`rpu_execution` 设置，将旧
`run.torch_num_threads` 转成 `cpu.threads`；GR00T/NavDP 的 seed 和步数/样本数保留。
LingBot 保留图片顺序和指令，将 `proprioception` 改为 `state`，旧 `input.seed` 优先
写入 `run.seed`；也可通过额外 `--input case=...` 或
`--input observation=...` 选择新输入。MoE 保留模型 profile 与显式长度，自动计算
文本文件的 `prompt_sha256`，并设置 `run.timing_mode="forward_argmax"`。

运行 MoE 命令前，将 `INPUT_IDS_SHA256` 设为对应 tokenizer 以
`add_special_tokens=False` 编码后、前 `prefill_tokens` 个连续 INT64 token ID 字节的
小写 SHA256。文本须足够长，且选取部分不含 EOS/PAD；输入构造逻辑见
[generation.py](../examples/common/generation.py)。

新配置通过配置检查后才写入新文件，已有输出会被拒绝。旧四图、embeddings-only 和
独立组件数据仍须按新模型准备；工具不转换这些输入文件。原 `runner.env` 和
`runner.torch_profile` 设置按当前模板另行检查。最后执行
`python examples/run.py --config /path/to/new-config.toml --check-config`，再按模型指南运行。

## 日志和转换变化

runner 默认 `logging.level="ERROR"`。需要显示生成文本和普通进度时，在 TOML 顶层
添加或修改：

```toml
[logging]
level = "INFO"
summary = true
```

`summary` 单独控制模型与运行摘要。日志等级及输出文件见
[示例日志说明](../examples/README.zh.md#logging-and-output)。

Wall-OSS W4 转换默认 `--group-size 32`，两个 decoder expert 的投影均使用 INT4，
包含 expert0 Down。逐通道 scale 使用 `--group-size 0`；非空 `--keep-int8` 会被拒绝。
按[完整命令](../examples/configs/WEIGHTS.md#cmd-wall)生成新输出目录。
分拣配方仍只从 W4 目录加载动作专家。

更换运行时资产、模型精度或 Graph 冷配置后，使用新进程。权重转换失败或只完成一部分时，
从未修改的原始权重重新构建。各参数生效时机和取值范围见[运行时配置](runtime_config.zh.md)。
