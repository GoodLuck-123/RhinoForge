'Wall action preprocessing fold with checkpoint weights.'
import json
import torch
import torch.nn.functional as F
from rpu_backend.model_registry import model_path
from rpu_backend.adapters.wall_oss.action import _sinusoidal_pos_emb
from rpu_backend.adapters.wall_oss.llm import _SafeTensorStore
_sin = _sinusoidal_pos_emb
_DEFAULT_CKPT = str(model_path("wall-oss-0.5"))


def _literal(t, x, dof, w1, w2, w3, AH):
    'CPU fp32 mirror of WallOssAction._action_embed (action.py:79-89).'
    na = torch.cat([x, dof], -1)
    ae = na @ w1.T
    te = _sin(torch.tensor([t]), AH).unsqueeze(1).repeat(1, x.size(1), 1)
    ce = torch.cat([ae, te], -1) @ w2.T
    return F.silu(ce) @ w3.T

def _load_weights():
    st = _SafeTensorStore(_DEFAULT_CKPT)
    g = lambda k: st.get_tensor(k).float()
    cfg = json.load(open(f"{_DEFAULT_CKPT}/config.json"))
    AH = cfg["action_hidden_size"]
    w1 = g("action_preprocessor.w1.weight")
    w2 = g("action_preprocessor.w2.weight")
    w3 = g("action_preprocessor.w3.weight")
    AD = g("action_preprocessor.action_proj_back.weight").size(0)
    return AH, AD, w1, w2, w3

def test_folded_preprocessor_matches_literal():
    AH, AD, w1, w2, w3 = _load_weights()
    H = 50
    torch.manual_seed(0)
    x = torch.randn(1, H, AD)
    dof = torch.randn(1, H, AD)
    W_comb = w2[:, :AH] @ w1[:, :AD]
    for t in (0.0, 0.3, 0.7, 1.0):
        ct = _sin(torch.tensor([t]), AH) @ w2[:, AH:].T
        B = (dof @ w1[:, AD:].T) @ w2[:, :AH].T + ct.unsqueeze(1)
        emb = F.silu(x @ W_comb.T + B) @ w3.T
        ref = _literal(t, x, dof, w1, w2, w3, AH)
        cos = F.cosine_similarity(emb.flatten(), ref.flatten(), 0).item()
        print(f"t={t} cos={cos:.8f}")
        assert cos > 0.9999, f"t={t} cos={cos}"
