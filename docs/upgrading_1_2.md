# Upgrade to v1.2.0

[简体中文](upgrading_1_2.zh.md) | English

This source tree targets v1.2.0. The [changelog](../CHANGELOG.md) summarizes the
changes; this guide covers installation and migration from the previous public
source. Package name `rhinoforge`, import `rpu_backend`, `torch.rpu`, and the
existing runtime-path environment variables remain unchanged.

## Install v1.2.0

1. Create a fresh Python 3.12 environment and build against the matching runtime
   assets using [Getting started](getting_started.md). Dependencies are declared
   in [pyproject.toml](../pyproject.toml), including PyTorch 2.10.0 and Transformers
   5.3.0. Select the `vision`, `vla` or `audio` extra for the corresponding models.
2. Obtain `RhinoForge-runtime-v1.2.0.tar.gz` and its checksum through the
   [asset request entry](runtime_assets.md#request-runtime-assets). Verify the
   package and retain both main and DDR-expansion assets with their sidecars.
   Rhino Launch remains version 1.0.0. Select the exact source and asset identities
   in [runtime-dependencies.json](../release/runtime-dependencies.json).
3. Update `RHINO_LAUNCH_ROOT`, `RPU_KERNEL_LIB_PATH` and
   `RPU_KERNEL_LIB_PATH_EXPANSION`, rebuild the extension, and start a new process.
   Keep the previous environment and asset set together for rollback.
4. Run `python examples/verify_install.py --check-config`, then validate the chosen
   TOML with `python examples/run.py --config PATH --check-config`. On an idle
   device, run that configuration once to check the installed path.

## Example entry points

Use `python examples/run.py --config PATH` for new commands. The previous model
scripts and `run_model.py` remain compatibility wrappers; `run_model.py` keeps
`--list-targets` and its Qwen3 default. GR00T, NavDP and standalone SigLIP callers
must select a configuration explicitly after the input migrations below.
The wrapper defaults and aliases are listed in
[legacy.py](../examples/common/legacy.py).

The following old paths remain compatibility aliases. New commands should use
the canonical paths; all paths below are relative to `examples/configs/`.

| Old path | Canonical path |
|---|---|
| `wall_oss/base/fp16.toml` | `wall_oss/legacy/fp16.toml` |
| `wall_oss/base/w8a16.toml` | `wall_oss/legacy/w8a16.toml` |
| `wall_oss/base/w4a16.toml` | `wall_oss/legacy/w4a16.toml` |
| `wall_oss/base/w4a16_pgrp.toml` | `wall_oss/legacy/w4a16_pgrp.toml` |
| `lingbot/v2/6b_w8a16.toml` | `lingbot/legacy/exact_z2/6b_w8a16.toml` |
| `rhinovla/v3/w8a16.toml` | `rhinovla/v3/expert_w8a16.toml` |

`rhinovla/v1/fp16.toml` retains its caller-owned runtime factory, checkpoint and
request contract. Use
[official_fp16.toml](../examples/configs/rhinovla/v1/official_fp16.toml) only with
the fixed official source, checkpoint and fixture in the
[v1 guide](../examples/configs/rhinovla/v1/README.md).

`g05/base/fp16.toml` retains the original G05 source-integration contract. Build
the official CPU `G05PolicyQwen35`, apply `patch_g05_policy_for_rpu`, and call its
continuous/discrete/CoT request path. The current K1 action examples require
separate assets and inputs; see the [G05 guide](../examples/configs/g05/README.md).

## Configurations requiring input migration

These old configuration contents cannot be used unchanged, even where the path
still exists. Select the matching checkpoint and prepare the inputs documented
by the current template.

| Old configuration | Required migration |
|---|---|
| `gr00t/n1_7_droid/fp16.toml` | The old four-image DROID/Cosmos profile and the current three-camera Qwen profile use different models and embodiment inputs. For a matching checkpoint, prepare the required camera/state tensors, map `request.input_pt` to `input.observation` and `request.seed/num_steps` to `input.seed/num_steps`. Follow the [GR00T guide](../examples/configs/gr00t/README.md). |
| `internvla/navdp/fp16.toml` | Prepare a full-chain `input.fixture` and `input.asset_manifest`; preserve `request.sample_num/seed` as `input.sample_num/seed`. The previous embeddings-only component file is insufficient. See [InternVLA](../examples/configs/internvla/README.md). |
| `siglip/pi05_component/fp16.toml` | Keep standalone component callers on the SigLIP adapter. To run Pi0.5, prepare its complete policy batch using the [Pi0.5 guide](../examples/configs/pi05/README.md). |
| `lingbot/v2/6b_fp16.toml` | Select camera count and language length, then provide `input.checkpoint`, `input.training_config` and matching inputs. Retain old images/instruction and map `proprioception` to `state`, or select a new `case` / `observation`. Preserve camera order, action dimensions and normalization. See [LingBot v2](../examples/configs/lingbot/v2/README.md); exact-Z2 W8 is a separate profile. |
| `qwen3_5/text/35b_a3b/mixed_e4m3_p4096_d4096.toml` | Replace synthetic `prompt_token_ids` / `prefill_fill_token_id` with `prompt_file`, `prompt_sha256` and `input_ids_sha256`; set `run.timing_mode="forward_argmax"`. Keep the checkpoint and explicit prefill/decode lengths, and record the new text input. See [Qwen3.5](../examples/configs/qwen3_5/README.md). |

## Migration tool

[`examples/migrate_config.py`](../examples/migrate_config.py) migrates an old TOML
to an explicitly selected current template. Prepare the complete inputs listed
above, then run the applicable command. Replace `/path/to/...` and source
filenames; the output parent directory must already exist.

```bash
# GR00T: supply the three-camera observation for the selected checkpoint.
python examples/migrate_config.py --source /path/to/old-gr00t.toml \
  --template examples/configs/gr00t/n1_7_droid/fp16.toml --output /path/to/new-gr00t.toml \
  --input observation=/path/to/three-camera-observation.pt

# NavDP: supply a complete-chain fixture and asset manifest.
python examples/migrate_config.py --source /path/to/old-navdp.toml \
  --template examples/configs/internvla/navdp/fp16.toml --output /path/to/new-navdp.toml \
  --input fixture=/path/to/fullchain-fixture.pt --input asset_manifest=/path/to/assets.json

# SigLIP to Pi0.5: explicitly select a complete policy with its batch/checkpoint.
python examples/migrate_config.py --source /path/to/old-siglip.toml \
  --template examples/configs/pi05/2cam/fp16.toml --output /path/to/new-pi05.toml \
  --input batch=/path/to/pi05-batch.pt --input checkpoint=/path/to/pi05-checkpoint

# LingBot: match cameras/language length with an FP16 template; retain old inputs.
python examples/migrate_config.py --source /path/to/old-lingbot.toml \
  --template examples/configs/lingbot/v2/3cam_t3/fp16.toml --output /path/to/new-lingbot.toml \
  --input checkpoint=/path/to/lingbot-checkpoint --input training_config=/path/to/training.yaml

# MoE: provide actual prose and the target token sequence's SHA256.
python examples/migrate_config.py --source /path/to/old-moe.toml \
  --template examples/configs/qwen3_5/text/35b_a3b/mixed_e4m3_p4096_d4096.toml \
  --output /path/to/new-moe.toml --input prompt_file=/path/to/prompt.txt \
  --input input_ids_sha256="$INPUT_IDS_SHA256"
```

The tool retains applicable `run`, `cpu`, `logging` and `rpu_execution` settings,
and maps `run.torch_num_threads` to `cpu.threads`. GR00T/NavDP seeds and step/sample
counts are retained. LingBot retains image order and instruction, maps
`proprioception` to `state`, and gives old `input.seed` precedence over `run.seed`. Add
`--input case=...` or `--input observation=...` to select a new input instead.
MoE preserves the model profile and explicit lengths, computes the text file's
`prompt_sha256`, and selects `run.timing_mode="forward_argmax"`.

Before the MoE command, set `INPUT_IDS_SHA256` to the lowercase SHA256 of the
first `prefill_tokens` contiguous INT64 token-ID bytes produced by the matching
tokenizer with `add_special_tokens=False`. The prose must contain enough tokens
and the selected sequence must exclude EOS/PAD. See
[generation.py](../examples/common/generation.py) for input construction.

The tool checks the generated configuration before creating a new file and
rejects existing outputs. Prepare new inputs for the old four-image,
embeddings-only and standalone-component cases; it does not convert those data
files. Review old `runner.env` and `runner.torch_profile` settings against the
selected template. Finally run
`python examples/run.py --config /path/to/new-config.toml --check-config`, then
execute it using the model guide.

## Logging and conversion changes

The runner defaults to `logging.level="ERROR"`. To show generated text and
ordinary progress messages, add or update this top-level table in your TOML:

```toml
[logging]
level = "INFO"
summary = true
```

`summary` separately controls the model and runtime summaries. Output files and
log levels are described in [example logging](../examples/README.md#logging-and-output).

Wall-OSS W4 conversion now uses `--group-size 32` by default and converts both
decoder experts, including expert-0 Down, to INT4. Use `--group-size 0` for
legacy per-channel scales; non-empty `--keep-int8` is rejected. Generate a new
output directory using the [complete commands](../examples/configs/WEIGHTS.md#cmd-wall).
The sorting recipe still loads only the action expert from that W4 directory.

Use a new process after changing runtime assets, model precision or cold Graph
settings. Rebuild from clean original weights after a failed or partial weight
transformation. See [runtime configuration](runtime_config.md) for each setting's
lifetime and supported range.
