"""Speech configuration and shared CLI contracts without board access."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples.common import cli, config
from examples.runners.qwen3_audio import runner

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ("qwen3_asr/0_6b/fp16.toml", "qwen3_tts/0_6b/base.toml",
             "qwen3_tts/0_6b/base_rpu_codec.toml")


@pytest.mark.parametrize("template", TEMPLATES)
def test_check_config_without_torch_or_backend(template):
    code = """
import sys
from examples.common.cli import main
assert main(['--config', sys.argv[1], '--check-config']) == 0
assert 'torch' not in sys.modules and 'rpu_backend' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(ROOT / "examples/configs" / template)],
                   cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("table,key,value", [
    ("qwen3_tts", "codec_backend", "cuda"), ("qwen3_tts", "source", ""),
    ("input", "text", "x" * 1025), ("input", "language", "Auto"),
    ("input", "max_new_tokens", True), ("run", "runs", 0),
    ("run", "warmup", -1), ("run", "hwperf", True),
    ("input", "unknown", 1), ("cpu", "threads", 0),
])
def test_invalid_speech_config(table, key, value):
    cfg = config.load_config(ROOT / "examples/configs" / TEMPLATES[1])
    cfg[table][key] = value
    with pytest.raises(ValueError):
        runner.validate_example_config(cfg)


def test_relative_paths_and_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "speech.toml"
    path.write_text('[qwen3_asr]\ncheckpoint="model"\n[input]\naudio="a.wav"\n[run]\noutput_dir="out"\n')
    cfg = config.load_config(path)
    assert cfg["input"]["audio"] == str(tmp_path / "a.wav")
    assert cfg["qwen3_asr"]["checkpoint"] == str(tmp_path / "model")
    assert config.config_target(cfg) == "qwen3_asr"
    seen = []
    monkeypatch.setattr(runner, "run_example_config", lambda value: seen.append(value) or 7)
    assert cli._dispatch(cfg, SimpleNamespace(check_config=False)) == 7
    assert seen == [cfg]


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("codec_backend", [None, "cpu", "rpu"])
def test_runner_records_results_and_closes_on_failure(tmp_path, monkeypatch, fail, codec_backend):
    import numpy as np
    import soundfile as sf
    from rpu_backend import api
    from examples.common import libraries
    from rpu_backend.api import qwen3_tts

    cfg = config.load_config(ROOT / "examples/configs" / TEMPLATES[2])
    if codec_backend is None:
        cfg["qwen3_tts"].pop("codec_backend")
    else:
        cfg["qwen3_tts"]["codec_backend"] = codec_backend
    cfg["run"].update(warmup=2, runs=3, output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(libraries, "current_libraries", lambda root: ({}, {}))
    monkeypatch.setattr(runner, "configure_cpu_threads", lambda value: {})
    monkeypatch.setattr(qwen3_tts, "_read_reference", lambda path: None)
    class Model:
        closed = False
        calls = 0
        def synthesize(self, *a, **kw):
            self.calls += 1
            if fail:
                raise RuntimeError("inference failed")
            return np.ones(24000, dtype=np.float32), 24000
        def execution_stats(self):
            return {"components": {"talker": {"graph_invariant_ok": True}}}
        def close(self):
            self.closed = True
    model = Model()
    def load(checkpoint, **kw):
        assert kw["codec_backend"] == (codec_backend or "rpu")
        return model
    monkeypatch.setattr(api, "Qwen3TTSModel", SimpleNamespace(from_pretrained=load))
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            runner.run_example_config(cfg)
    else:
        assert runner.run_example_config(cfg) == 0
        assert model.calls == 5
        wave, rate = sf.read(tmp_path / "out/output.wav")
        assert rate == 24000 and len(wave) == 24000
    assert model.closed
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["execution"]["status"] == ("FAILED" if fail else "COMPLETED")
    if not fail:
        assert len(report["wall_ms_runs"]) == 3 and len(report["warmup_wall_ms_runs"]) == 2
        assert report["median_rtf"] == report["median_ms"] / 1000


def test_speech_summary_uses_rtf_not_token_rate():
    from examples.common.performance import summary
    lines = summary({"qwen3_tts": {}}, {"execution": {"status": "COMPLETED"}, "median_rtf": 0.9})
    assert lines == ["TTS RTF median=0.900; RTF = request time / audio duration"]
