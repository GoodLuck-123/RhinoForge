from __future__ import annotations
import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap
from rpu_backend.tests.acceptance import COLD_IMPORT_TIMEOUT_S, timeout_message
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

def test_import_defers_qwen3_adapter_and_class_patches():
    probe = textwrap.dedent('\n        import sys\n        from transformers.models.qwen3.modeling_qwen3 import (\n            Qwen3RMSNorm,\n            Qwen3RotaryEmbedding,\n        )\n\n        assert "rpu_backend" not in sys.modules\n        assert not getattr(Qwen3RMSNorm, "_rpu_patched_rmsnorm", False)\n        assert not getattr(Qwen3RotaryEmbedding, "_rpu_patched_rotary", False)\n        rmsnorm_forward = Qwen3RMSNorm.forward\n        rotary_forward = Qwen3RotaryEmbedding.forward\n\n        import rpu_backend\n        from rpu_backend.runtime.registry import get_adapter, list_adapters\n\n        assert rpu_backend.__all__ == [\n            "__version__",\n            "RPUCache",\n            "RPUModelForCausalLM",\n            "reset_graph_cache",\n        ]\n        assert "freeze_for_dynamo" not in vars(rpu_backend)\n        assert "rpu_backend.graph.lazy_init_guard" not in sys.modules\n        builtins = {\n            "Qwen3ForCausalLM",\n            "LlamaForCausalLM",\n            "DINOv3ViTModel",\n            "Gemma4ForConditionalGeneration",\n            "Qwen3_5ForConditionalGeneration",\n            "Qwen3VLForConditionalGeneration",\n        }\n        assert builtins <= set(list_adapters())\n        for module in (\n            "rpu_backend.adapters.qwen3",\n            "rpu_backend.adapters.llama",\n            "rpu_backend.adapters.dinov3",\n            "rpu_backend.adapters.gemma4",\n            "rpu_backend.adapters.qwen3_5",\n            "rpu_backend.adapters.qwen3_vl",\n        ):\n            assert module not in sys.modules\n        assert Qwen3RMSNorm.forward is rmsnorm_forward\n        assert Qwen3RotaryEmbedding.forward is rotary_forward\n        assert not getattr(Qwen3RMSNorm, "_rpu_patched_rmsnorm", False)\n        assert not getattr(Qwen3RotaryEmbedding, "_rpu_patched_rotary", False)\n\n        assert callable(rpu_backend.freeze_for_dynamo)\n        assert "rpu_backend.graph.lazy_init_guard" in sys.modules\n        assert "freeze_for_dynamo" not in vars(rpu_backend)\n\n        adapter = get_adapter("Qwen3ForCausalLM")\n        assert adapter.__name__ == "Qwen3Adapter"\n        assert "rpu_backend.adapters.qwen3" in sys.modules\n        assert getattr(Qwen3RMSNorm, "_rpu_patched_rmsnorm", False)\n        assert getattr(Qwen3RotaryEmbedding, "_rpu_patched_rotary", False)\n\n        from rpu_backend.api import Pi05Policy\n\n        class FakeLeRobotPolicy:\n            def __init__(self):\n                self.calls = []\n\n            def predict_action_chunk(self, batch, **kwargs):\n                self.calls.append((batch, kwargs))\n                return "chunk"\n\n        policy = Pi05Policy.__new__(Pi05Policy)\n        policy._rpu_ready = True\n        policy._lerobot_policy = FakeLeRobotPolicy()\n        assert policy.predict_action_chunk({"x": 1}, num_steps=7) == "chunk"\n        assert policy._lerobot_policy.calls == [({"x": 1}, {"num_steps": 7})]\n        ')
    env = os.environ.copy()
    env.pop('RPU_BACKEND_SO', None)
    env['PYTHONPATH'] = os.pathsep.join([str(REPO_ROOT / 'python'), *(p for p in sys.path if p)])
    try:
        result = subprocess.run([sys.executable, '-S', '-c', probe], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=COLD_IMPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(timeout_message('lazy-adapter probe', exc)) from exc
    assert result.returncode == 0, f'lazy-adapter probe failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
