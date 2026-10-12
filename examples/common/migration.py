"""Convert the five former input schemas to an explicitly selected template."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import tomllib

from examples.common.config import ConfigError, config_target, load_config
from examples.common.legacy import resolve_config_path


def migration_kind(config):
    for name in ("runner", "example", "input", "request", "run", "cpu", "logging", "rpu_execution"):
        if name in config and not isinstance(config[name], dict):
            raise ConfigError(f"[{name}] must be a table")
    target = config.get("runner", {}).get("target")
    if target in ("gr00t", "internvla_navdp", "siglip"):
        return target
    example = config.get("example", {})
    if example.get("profile_id") == "lingbot-vla-v2-6b.fp16.source-only":
        return "lingbot_fp16"
    if (example.get("architecture") == "Qwen3_5MoeForConditionalGeneration"
            and {"prompt_token_ids", "prefill_fill_token_id"} & config.get("input", {}).keys()):
        return "moe_synthetic"
    raise ConfigError("this configuration does not use one of the five legacy input schemas")


def _required(inputs, names):
    missing = [name for name in names if not isinstance(inputs.get(name), str) or not inputs[name].strip()]
    if missing:
        raise ConfigError("provide --input for: " + ", ".join(missing))


def migrate(config, template, inputs):
    """Preserve compatible settings; new model/input selection comes from the caller."""
    kind = migration_kind(config)
    target = {"gr00t": "gr00t", "internvla_navdp": "internvla_n1", "siglip": "pi05",
              "lingbot_fp16": "lingbot2", "moe_synthetic": "qwen3_5"}[kind]
    if config_target(template) != target:
        raise ConfigError(f"{kind} migration requires a current {target} template")
    result = deepcopy(template)
    result.pop("_config_path", None)
    result.pop("_config_dir", None)
    new_input = result.setdefault("input", {})
    old_input = config.get("input", {})
    request = config.get("request", {})

    if kind == "gr00t":
        _required(inputs, ("observation",))
        for key in ("seed", "num_steps"):
            if key in request:
                new_input[key] = request[key]
    elif kind == "internvla_navdp":
        _required(inputs, ("fixture", "asset_manifest"))
        for key in ("seed", "sample_num"):
            if key in request:
                new_input[key] = request[key]
    elif kind == "siglip":
        _required(inputs, ("batch", "checkpoint"))
    elif kind == "lingbot_fp16":
        _required(inputs, ("checkpoint", "training_config"))
        if result.get("lingbot2", {}).get("precision") != "fp16":
            raise ConfigError("the former LingBot FP16 configuration requires an FP16 template")
        for key in ("case", "observation", "images", "instruction", "state", "noise",
                    "expected_language_tokens"):
            new_input.pop(key, None)
        if not {"case", "observation", "images"} & inputs.keys():
            for old, new in (("images", "images"), ("instruction", "instruction"),
                             ("proprioception", "state")):
                if old in old_input:
                    new_input[new] = deepcopy(old_input[old])
            if isinstance(new_input.get("images"), list):
                new_input["cameras"] = len(new_input["images"])
    else:
        if result.get("example", {}) != config.get("example", {}):
            raise ConfigError("MoE migration must retain the checkpoint profile, dtype and architecture")
        _required(inputs, ("prompt_file", "input_ids_sha256"))
        for key in ("prefill_tokens", "decode_steps", "stop_on_eos"):
            if key in old_input:
                new_input[key] = old_input[key]
        if not re.fullmatch(r"[0-9a-f]{64}", inputs["input_ids_sha256"]):
            raise ConfigError("input_ids_sha256 must be the lowercase SHA256 of the token IDs")
        prompt = Path(inputs["prompt_file"]).expanduser().resolve(strict=True)
        digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
        if "prompt_sha256" in inputs and inputs["prompt_sha256"] != digest:
            raise ConfigError("prompt_sha256 differs from the selected text file")
        inputs = dict(inputs, prompt_file=str(prompt), prompt_sha256=digest)

    new_input.update(deepcopy(inputs))
    # Model geometry and input contracts come from the selected current template.
    for section in ("run", "cpu", "logging", "rpu_execution"):
        if section in config:
            _merge(result.setdefault(section, {}), config[section])
    run = result.setdefault("run", {})
    if "torch_num_threads" in run:
        result.setdefault("cpu", {})["threads"] = run.pop("torch_num_threads")
    if kind == "moe_synthetic":
        run["timing_mode"] = "forward_argmax"
    if kind == "lingbot_fp16" and "seed" in old_input:
        run["seed"] = old_input["seed"]
    return result


def _merge(target, source):
    if not isinstance(source, dict):
        raise ConfigError("configuration sections must be tables")
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _value(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int) or (isinstance(value, float) and math.isfinite(value)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    raise ConfigError(f"unsupported TOML value type: {type(value).__name__}")


def dumps(config):
    lines = []
    def table(data, path):
        if path:
            lines.append("[" + ".".join(json.dumps(key) for key in path) + "]")
        for key, value in data.items():
            if not isinstance(value, dict):
                lines.append(f"{json.dumps(key)} = {_value(value)}")
        lines.append("")
        for key, value in data.items():
            if isinstance(value, dict):
                table(value, (*path, key))
    table(config, ())
    return "\n".join(lines).lstrip().rstrip() + "\n"


def _input(value):
    key, sep, raw = value.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not raw:
        raise argparse.ArgumentTypeError("use KEY=VALUE for one [input] field")
    try:
        parsed = tomllib.loads("value=" + raw)["value"]
    except tomllib.TOMLDecodeError:
        parsed = raw
    if key.endswith("sha256") and re.fullmatch(r"[0-9a-f]{64}", raw):
        parsed = raw
    return key, parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Migrate an old example to a selected current template.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", action="append", type=_input, default=[], metavar="KEY=VALUE",
                        help="Set a new input field; quote TOML strings containing special characters.")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ConfigError("output already exists; choose a new file")
        inputs = dict(args.input)
        if len(inputs) != len(args.input):
            raise ConfigError("each --input key may be supplied only once")
        source = tomllib.loads(args.source.read_text(encoding="utf-8"))
        # Validate the chosen template before retaining its literal, serializable fields.
        args.template = resolve_config_path(args.template)
        load_config(args.template)
        template = tomllib.loads(args.template.read_text(encoding="utf-8"))
        converted = migrate(source, template, inputs)
        content = dumps(converted)
        with tempfile.TemporaryDirectory() as directory:
            check = Path(directory) / "migrated.toml"
            check.write_text(content, encoding="utf-8")
            load_config(check)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
        print(f"migrated configuration: {args.output}")
        runner = source.get("runner", {})
        if runner.get("env") or runner.get("torch_profile"):
            print("The selected template owns runtime settings; review the old [runner.env] "
                  "and [runner.torch_profile] settings separately.")
        print("Configuration checks passed. The selected runtime validates input assets when run.")
        return 0
    except (ConfigError, OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
