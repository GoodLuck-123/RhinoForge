import sys
from pathlib import Path
import torch
_QUANT = Path(__file__).resolve().parents[2] / 'quant'
sys.path.insert(0, str(_QUANT))
import convert_pi05 as _cp
_bits_for_name = _cp._bits_for_name
_convert_tensor = _cp._convert_tensor
_quant_config = _cp._quant_config
VLM = '.paligemma_with_expert.paligemma.model.language_model.layers.'
KEEP = ('k_proj.weight', 'v_proj.weight')

def _name(suffix: str) -> str:
    return f'model{VLM}0.{suffix}'

def test_bits_for_name_keeps_kv_int8_under_fake_w4():
    assert _bits_for_name(_name('self_attn.k_proj.weight'), 4, KEEP) == 8
    assert _bits_for_name(_name('self_attn.v_proj.weight'), 4, KEEP) == 8
    assert _bits_for_name(_name('self_attn.q_proj.weight'), 4, KEEP) == 4
    assert _bits_for_name(_name('self_attn.o_proj.weight'), 4, KEEP) == 4
    assert _bits_for_name(_name('mlp.down_proj.weight'), 4, KEEP) == 4

def test_bits_for_name_noop_without_keep_or_when_int8_default():
    assert _bits_for_name(_name('self_attn.k_proj.weight'), 4, ()) == 4
    assert _bits_for_name(_name('self_attn.k_proj.weight'), 8, KEEP) == 8

def test_convert_tensor_value_ranges_match_selected_bits():
    torch.manual_seed(0)
    W = torch.randn(64, 128).to(torch.float16)
    name = _name('self_attn.k_proj.weight')
    (out4, nq, _, _) = _convert_tensor(name, W, bits=4, int8_keep_suffixes=())
    assert nq == 1 and out4[name].dtype == torch.int8
    assert int(out4[name].max()) <= 7 and int(out4[name].min()) >= -8
    (out8, _, _, _) = _convert_tensor(name, W, bits=4, int8_keep_suffixes=KEEP)
    assert out8[name].dtype == torch.int8
    assert int(out8[name].abs().max()) > 7
    qname = _name('self_attn.q_proj.weight')
    (outq, _, _, _) = _convert_tensor(qname, W, bits=4, int8_keep_suffixes=KEEP)
    assert int(outq[qname].max()) <= 7 and int(outq[qname].min()) >= -8

def test_quant_config_records_mixed_suffixes():
    cfg = _quant_config(bits=4, int8_keep_suffixes=KEEP)
    assert cfg['method'] == 'w4a16_fake_int8'
    assert cfg['value_bits'] == 4
    assert cfg['mixed_int8_suffixes'] == list(KEEP)
