# Run and debug models

[简体中文](model_testing.zh.md) | English

Select a TOML from [model examples](../examples/README.md). Runners live in `examples/runners/`; shared configuration and execution code is in `examples/common/`.

## Run

```sh
python -m examples.run --config examples/configs/qwen3/text/0_6b/fp16.toml --check-config
python -m examples.run --config examples/configs/qwen3_vl/vl/4b/w8a16_text_head_vision.toml
python -m examples.run --config examples/configs/pi05/2cam/w8a16_action_nvfp4.toml
```

Replace placeholder paths and supply the checkpoint and inputs required by the model guide. Relative `input.prompt_file` and `input.prepared_input_safetensors` paths resolve from the configuration file's directory. Other asset paths follow the model guide. `--check-config` validates configuration without loading weights or accessing a device. Quantized checkpoints require the loader's metadata; see [quantization](quantization.md).

Preprocessed-input examples accept CPU tensor mappings produced by the checkpoint's processor. Follow the selected model's shape and component requirements.

## Execution controls

- `run.warmup` defaults to two calls: BUILD followed by REPLAY for a new Graph signature. Set `run.runs` for subsequent calls.
- With `warmup >= 2`, omit `warmup_decode_steps` or match `input.decode_steps`. Explicit `warmup=0` or `1` allows shortened cold-path diagnostics.
- Decode steps count calls after the prefill token. EOS may stop generation early; fixed-length runs use `stop_on_eos=false` in both phases.
- `run.output_dir` selects the output directory. Check the generated files for outputs and Graph state.

## Diagnose a call

`--profile` or `[run].profile=true` adds one Torch profiler call after ordinary execution. Enable shapes, memory or stacks only when needed. Traces can contain application data; store them outside the source tree.

The public runner rejects `hwperf=true`. For a device-specific issue, use the installed runtime's profiler and record the matching asset and input shape. See [runtime configuration](runtime_config.md) for controls and [validation](validation_policy.md) for focused checks.

## Qwen3 speech

Install the `audio` extra and call `Qwen3ASRModel` or `Qwen3TTSModel`.
The [speech guide](qwen3_audio.md) provides checkpoint downloads, TTS source
preparation and complete Python examples.

Speech CLI: `python -m examples.run --config examples/configs/qwen3_tts/0_6b/base.toml --check-config`; fill local paths before running. [ASR templates](../examples/configs/qwen3_asr/README.md) and [TTS templates](../examples/configs/qwen3_tts/README.md) record resident latency, process CPU time and RTF.

## Repository checks

Run the host suite with `make ci-local-core PYTHON=python`. It uses the same
host paths and device exclusions as CI. `make ci-local-quickstart` checks
example configurations and the adapter registry.

After configuring a source build with the [public SDK](getting_started.md),
enable the optional C++ contracts in that build directory:

```sh
cmake -S . -B build -DRHINOFORGE_BUILD_HOST_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
cmake -S . -B build -DRHINOFORGE_BUILD_DEVICE_TESTS=ON
cmake --build build --target rhino_device_tests
```

Device executables are in `build/tests/`; run them on an idle device with the
configured runtime assets. Both options default to OFF. Device tests are
separate from the host CI suite.
