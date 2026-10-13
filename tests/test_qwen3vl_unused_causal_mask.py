"""Execute the public mask guard/transport boundary without native hardware."""
import ast
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters/qwen3_vl/__init__.py"


def _boundary(mask):
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rpu_qwen3vl_forward")
    # Execute the real top-level caller-mask validation and the real final
    # native-mask assignment. Earlier CPU position preparation retains mask.
    guard = next(n for n in fn.body if isinstance(n, ast.If) and
                 ast.unparse(n.test) == "attention_mask is not None")
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == "attn_rpu" for t in n.targets))
    ns = {"torch": torch, "attention_mask": mask, "input_ids": torch.ones(1, 3, dtype=torch.long),
          "self": object()}
    exec(compile(ast.Module(body=[guard, assignment], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def test_all_valid_mask_stays_available_for_cpu_positions_without_rpu_upload(monkeypatch):
    mask = torch.ones(1, 3, dtype=torch.long)
    real_to = torch.Tensor.to
    def forbid_upload(tensor, *args, **kwargs):
        if (args and str(args[0]) == "rpu") or str(kwargs.get("device")) == "rpu":
            pytest.fail("causal consumer must not upload caller mask")
        return real_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", forbid_upload)
    result = _boundary(mask)
    assert result["attention_mask"] is mask
    assert result["attn_rpu"] is None
    assert _boundary(None)["attn_rpu"] is None


@pytest.mark.parametrize("mask,error", [(torch.tensor([[1, 0, 1]]), NotImplementedError),
    (torch.ones(1, 1, 3, 3), NotImplementedError), (torch.ones(1, 2), ValueError)])
def test_causal_mask_removal_does_not_accept_unsupported_semantics(mask, error):
    with pytest.raises(error):
        _boundary(mask)
