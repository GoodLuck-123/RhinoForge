"""LingBot-VLA V2 — prefix composition / M-RoPE / theta: PURE-CPU tests. NEEDS NO device.

Covers the two integration gaps closed without the router/oplib escalation:
  1. exact `embed_prefix` reproduction (vision boundaries + the 8+8 align query tokens),
     token order, S, and the align spans for text-only / 1-image / multi-image;
  2. p0 vs S divergence, position_ids vs an INDEPENDENT CPU re-implementation of the official
     `build_prefix_position_ids`, theta=5e6 plumbing, and the M-RoPE → 1D reduction.

Nothing here imports `rpu_backend` (loading the .so touches /dev/rpu). `prefix.py` is exec'd
straight from its file path, and the runtime is parsed with `ast` rather than imported.

Set ``RPU_MODEL_CACHE`` when running the optional real-weight assertion.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ADAPTER = (Path(__file__).resolve().parents[2]
           / "python" / "rpu_backend" / "adapters" / "lingbot_vla_v2")
REPO_ROOT = Path(__file__).resolve().parents[2]
PREFIX_PY = ADAPTER / "prefix.py"
RUNTIME_PY = ADAPTER / "runtime.py"
QWEN3VL_VISION_PY = (ADAPTER.parent / "qwen3_vl" / "vision.py")
CONVERT_PY = ADAPTER / "convert.py"
QWEN3VL_BASE = ADAPTER / "_qwen3vl_base"
RPU_REGISTRATIONS = REPO_ROOT / "src" / "core" / "rpu_dispatch_registrations.inc"
MODEL_REGISTRY_PY = REPO_ROOT / "python" / "rpu_backend" / "model_registry.py"


def _registered_model_path() -> Path:
    spec = importlib.util.spec_from_file_location("lingbot2_prefix_model_registry", MODEL_REGISTRY_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.model_path("lingbot-vla-v2-6b")


WEIGHTS = _registered_model_path()


def _exec_standalone(path: Path, name: str):
    """Exec a module from its file path — bypasses rpu_backend/__init__.py (no device touch).

    The module MUST be registered in sys.modules before exec_module: `dataclasses` resolves a
    class's annotations via `sys.modules[cls.__module__]`, so a frozen dataclass in an
    unregistered module raises during class creation.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def px():
    return _exec_standalone(PREFIX_PY, "lb2_prefix_cpu")


@pytest.fixture(scope="module")
def cfg():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    if not Path(QWEN3VL_BASE).exists():
        pytest.skip("Qwen3-VL-4B-Instruct base config not present")
    return Qwen3VLConfig.from_pretrained(QWEN3VL_BASE)


@pytest.fixture(scope="module")
def ids(px, cfg):
    return px.SpecialTokenIds.from_config(cfg)


@pytest.fixture(scope="module")
def acfg(px):
    return px.AlignConfig()          # defaults == the authoritative kuavo_v2_depth YAML


@pytest.fixture(scope="module")
def rope_model(cfg):
    """Meta-init Qwen3VLModel — get_rope_index is pure, so no weights are ever materialised."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    with torch.device("meta"):
        return Qwen3VLModel(cfg)


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [PREFIX_PY, CONVERT_PY])
def test_module_is_cpu_safe(path):
    """prefix.py / convert.py must never import rpu_backend at module scope, nor use
    relative imports — that is what lets these tests exec them on an device-less host."""
    tree = ast.parse(path.read_text())
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] == "rpu_backend"]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad.append(f"relative import level={node.level}")
            elif node.module and node.module.split(".")[0] == "rpu_backend":
                bad.append(node.module)
    assert not bad, f"{path.name} must stay device-free at module scope; found: {bad}"


# ---------------------------------------------------------------------------
# Align config / segment order
# ---------------------------------------------------------------------------
def test_segment_order_matches_yaml(px, acfg):
    """share_future_depth_query=True ⇒ future_video contributes NO own segment."""
    assert px.prefix_query_segments(acfg) == ("language", "current_depth", "future_depth")
    assert px.num_query_tokens(acfg) == 16          # 8 current_depth + 8 future_depth


# ---------------------------------------------------------------------------
# Finite REAL-prefix execution planning / adapter-owned tail padding
# ---------------------------------------------------------------------------
def test_generic_prefix_plan_covers_max_72_language_tokens(px):
    # Exact three-camera geometry: 3 * (64 visual + 2 boundaries) +
    # 72 language + 16 align queries = REAL 286.  Generic execution aligns only
    # the tail; it does not truncate the public max_lang_tokens=72 contract.
    plan = px.plan_prefix_execution(
        286, max_seq=1024, suffix_len=51, chunk_size=240,
        padding_multiple=16,
    )
    assert (plan.real_len, plan.execution_len, plan.chunk_size) == (286, 288, 240)
    assert plan.key == (288, 240)


def test_fixed_z2_prefix_plan_is_a_bounded_real_envelope(px):
    plans = {
        px.plan_prefix_execution(
            real_len, max_seq=1024, suffix_len=51, chunk_size=0,
            fixed_execution_len=225,
        ).key
        for real_len in range(215, 226)
    }
    assert plans == {(225, 0)}
    with pytest.raises(ValueError, match="REAL prefix exceeds"):
        px.plan_prefix_execution(
            226, max_seq=1024, suffix_len=51, chunk_size=0,
            fixed_execution_len=225,
        )


def test_execution_padding_is_after_language_and_align_rows(px, acfg, ids):
    lang = torch.arange(1000, 1005, dtype=torch.long).view(1, -1)
    real = px.build_prefix_layout(3, 64, lang, acfg, ids)
    assert real.S == 219
    padded = px.pad_prefix_layout_for_execution(
        real, 225, pad_token_id=ids.eos
    )
    assert padded.real_len == 219 and padded.execution_len == 225
    assert padded.real.spans["future_depth"] == (211, 219)
    assert torch.equal(padded.input_ids[:, :219], real.input_ids)
    assert padded.pad_masks[:, :219].all()
    assert not padded.pad_masks[:, 219:].any()
    assert not padded.visual_pos_masks[:, 219:].any()
    assert padded.mm_token_type_ids[:, 219:].eq(px.MM_TYPE_TEXT).all()


def test_execution_padding_does_not_change_suffix_position_start(px):
    real_pos = torch.arange(12, dtype=torch.long).view(1, 1, 12).expand(3, 1, 12)
    padded_pos = torch.cat(
        (real_pos, torch.full((3, 1, 4), 999, dtype=torch.long)), dim=2
    )
    real_mask = torch.ones(1, 12, dtype=torch.bool)
    padded_mask = torch.cat(
        (real_mask, torch.zeros(1, 4, dtype=torch.bool)), dim=1
    )
    assert px.suffix_position_start(real_pos, real_mask) == 12
    assert px.suffix_position_start(padded_pos, padded_mask) == 12


def test_align_config_validation_mirrors_official(px):
    with pytest.raises(ValueError, match="share_future_depth_query"):
        px.AlignConfig(use_future_depth=False)              # sharing needs future_depth
    with pytest.raises(ValueError, match="use_current_patch_loss"):
        px.AlignConfig(use_current_video_patch=False)       # shared proj needs current patch
    with pytest.raises(ValueError, match="query depth alignment"):
        px.AlignConfig(align_type="mlp")
    with pytest.raises(ValueError, match="divisible"):
        px.AlignConfig(num_backbone_tokens=250)


def test_future_video_span_is_empty_under_sharing(px, acfg):
    """_future_video_own_token_count == 0 ⇒ _block_suffix_to_future_video_ is a NO-OP."""
    s, e = px.future_video_query_span(300, acfg)
    assert s == e, "future-video owns no tokens when it shares the future-depth query"


def test_future_video_span_nonempty_without_sharing(px):
    """Guard the OTHER branch: turning sharing off gives future_video its own 8 tokens."""
    c = px.AlignConfig(future_video_share_future_depth_query=False,
                       use_shared_future_task_proj=False)
    assert px.prefix_query_segments(c) == ("language", "current_depth", "future_video", "future_depth")
    s, e = px.future_video_query_span(300, c)
    assert (s, e) == (300 - 8 - 8, 300 - 8)


# ---------------------------------------------------------------------------
# Prefix layout: token order + S
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_images,num_patch,lang_len", [
    (1, 256, 8),      # one 32x32-patch image
    (1, 64, 12),      # one 16x16-patch image
    (3, 64, 5),       # multi-image
])
def test_prefix_layout_token_order_and_S(px, acfg, ids, n_images, num_patch, lang_len):
    lang = torch.arange(1000, 1000 + lang_len, dtype=torch.long).view(1, -1)
    lay = px.build_prefix_layout(n_images, num_patch, lang, acfg, ids)

    img_block = num_patch + 2                                   # + vision_start/end
    assert lay.image_token_len == img_block
    assert lay.S == n_images * img_block + lang_len + 16
    assert lay.input_ids.shape == (1, lay.S)

    row = lay.input_ids[0]
    for i in range(n_images):
        b = i * img_block
        assert row[b] == ids.vision_start
        assert row[b + img_block - 1] == ids.vision_end
        assert torch.all(row[b + 1: b + 1 + num_patch] == ids.image)
    a, b = lay.spans["language"]
    assert torch.equal(row[a:b], lang[0])
    for seg in ("current_depth", "future_depth"):
        a, b = lay.spans[seg]
        assert b - a == 8
        assert torch.all(row[a:b] == ids.eos), "align queries carry FAKE eos ids"
    # current_depth precedes future_depth; future_depth is the TAIL (tail blocking depends on it)
    assert lay.spans["current_depth"][1] == lay.spans["future_depth"][0]
    assert lay.spans["future_depth"][1] == lay.S


def test_zero_image_prefix_layout(px, acfg, ids):
    """A 0-image (text-only) prefix: no image block, but the 16 align queries still ride.

    This is the case where p0 == S holds (see test_p0_equals_S_text_only), which is exactly
    what the legacy-.so path (no cos_sin_offset arg) is restricted to.
    """
    lang = torch.arange(1000, 1008, dtype=torch.long).view(1, -1)
    lay = px.build_prefix_layout(0, 256, lang, acfg, ids)
    assert lay.S == 8 + 16                      # no image block at all
    assert lay.spans["image"] == (0, 0)
    assert not lay.visual_pos_masks.any()


def test_mm_token_type_ids_mark_only_image_placeholders(px, acfg, ids):
    lang = torch.arange(1000, 1005, dtype=torch.long).view(1, -1)
    lay = px.build_prefix_layout(2, 64, lang, acfg, ids)
    # type 1 exactly where the id is <image>; vision_start/end are TEXT (type 0)
    assert torch.equal(lay.mm_token_type_ids[0] == 1, lay.input_ids[0] == ids.image)
    assert int((lay.mm_token_type_ids == 1).sum()) == 2 * 64
    assert torch.equal(lay.visual_pos_masks, lay.mm_token_type_ids == 1)


def test_visual_mask_excludes_boundaries(px, acfg, ids):
    """deepstack scatter must target the <image> run only — never vision_start/end."""
    lang = torch.arange(1000, 1003, dtype=torch.long).view(1, -1)
    lay = px.build_prefix_layout(1, 64, lang, acfg, ids)
    v = lay.visual_pos_masks[0]
    assert not v[0] and not v[65]           # boundaries excluded
    assert v[1:65].all()                    # the 64 patches


# ---------------------------------------------------------------------------
# Align token math
# ---------------------------------------------------------------------------
def test_get_align_tokens_pooling_order(px):
    """`view(num_task_tokens, N//ntt, D).mean(1)` — token i = mean of rows [i*32,(i+1)*32).

    The transposed reading (view(N//ntt, ntt, D)) is the plausible wrong one; pin it down.
    """
    embs = torch.arange(256 * 4, dtype=torch.float32).view(256, 4)
    got = px.get_align_tokens(embs, 8)
    assert got.shape == (8, 4)
    assert torch.allclose(got[0], embs[0:32].mean(0))
    assert torch.allclose(got[7], embs[224:256].mean(0))
    with pytest.raises(ValueError):
        px.get_align_tokens(embs, 7)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="v2-6b weights not present")
def test_compute_align_tokens_against_real_weights(px, acfg):
    """Real checkpoint: shapes + that BOTH shared_task_proj projections actually fire."""
    cv = _exec_standalone(CONVERT_PY, "lb2_convert_align")
    gw, _ = cv.ckpt_reader(str(WEIGHTS))
    align_W = cv.load_align_weights(gw)
    toks = px.compute_align_tokens(align_W, acfg)
    assert set(toks) == {"current_depth", "future_depth"}
    for k, v in toks.items():
        assert v.shape == (8, 2560), f"{k}: {tuple(v.shape)}"
        assert torch.isfinite(v).all()
    # A projection genuinely ran: the result must differ from the bare mean-pooled query.
    bare = px.get_align_tokens(align_W["model.depth_align_embs"], 8)
    assert not torch.allclose(toks["current_depth"], bare), "current_shared_task_proj did not fire"


# ---------------------------------------------------------------------------
# position_ids / p0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_images,g,lang_len", [(1, 32, 8), (2, 16, 4)])
def test_position_ids_match_official_recipe(px, acfg, ids, rope_model, n_images, g, lang_len):
    """Our get_rope_index call reproduces the official build_prefix_position_ids recipe.

    Independent check: rebuild position ids from the raw HF call with the same arguments the
    official passes (prefix_input_ids, pad_masks.long(), one grid row per valid image).
    """
    sm = 2
    num_patch = (g // sm) ** 2
    lang = torch.arange(1000, 1000 + lang_len, dtype=torch.long).view(1, -1)
    lay = px.build_prefix_layout(n_images, num_patch, lang, acfg, ids)
    grid = torch.tensor([[1, g, g]] * n_images, dtype=torch.long)

    pos, _ = rope_model.get_rope_index(
        input_ids=lay.input_ids, mm_token_type_ids=lay.mm_token_type_ids,
        image_grid_thw=grid, video_grid_thw=None, attention_mask=lay.pad_masks.long())
    assert pos.shape == (3, 1, lay.S)

    # Independent expectation: text advances 1/token; each image advances max(h,w)//sm.
    cur = 0
    img_block = num_patch + 2
    for i in range(n_images):
        assert int(pos[0, 0, i * img_block]) == cur          # <vision_start> is TEXT
        cur += 1
        # the <image> run occupies ONE contiguous vision block
        cur += max(g, g) // sm
        assert int(pos[0, 0, i * img_block + img_block - 1]) == cur   # <vision_end> is TEXT
        cur += 1
    # language + 16 align tokens are all TEXT -> strictly sequential from here
    tail_start = n_images * img_block
    tail = pos[0, 0, tail_start:]
    assert torch.equal(tail, torch.arange(cur, cur + lay.S - tail_start))
    # all three axes agree on the TEXT tail (this is what makes 1-D RoPE exact for the suffix)
    assert torch.equal(pos[0, 0, tail_start:], pos[1, 0, tail_start:])
    assert torch.equal(pos[1, 0, tail_start:], pos[2, 0, tail_start:])


def test_p0_diverges_from_S_with_images(px, acfg, ids, rope_model):
    """THE blocker's premise, pinned numerically: p0 != S whenever the prefix holds images."""
    g, sm = 32, 2
    num_patch = (g // sm) ** 2                                     # 256
    lang = torch.arange(1000, 1008, dtype=torch.long).view(1, -1)
    lay = px.build_prefix_layout(1, num_patch, lang, acfg, ids)
    grid = torch.tensor([[1, g, g]], dtype=torch.long)
    pos, _ = rope_model.get_rope_index(
        input_ids=lay.input_ids, mm_token_type_ids=lay.mm_token_type_ids,
        image_grid_thw=grid, video_grid_thw=None, attention_mask=lay.pad_masks.long())
    p0 = px.suffix_position_start(pos, lay.pad_masks)
    # S = 256 patches + 2 boundaries + 8 lang + 16 align = 282
    assert lay.S == 282
    # p0: <vision_start>@0, image spans 16 (=32//2) -> 1..16, <vision_end>@17, then 24 text
    # tokens (8 lang + 16 align) -> 18..41 => max 41 => p0 = 42
    assert p0 == 42
    assert p0 != lay.S and lay.S - p0 == 240


def test_p0_equals_S_text_only(px, acfg, ids, rope_model):
    """The legacy-.so path is only correct when p0 == S — true iff there is no image."""
    lang = torch.arange(1000, 1008, dtype=torch.long).view(1, -1)
    lay = px.build_prefix_layout(0, 256, lang, acfg, ids)
    pos, _ = rope_model.get_rope_index(
        input_ids=lay.input_ids, mm_token_type_ids=lay.mm_token_type_ids,
        image_grid_thw=None, video_grid_thw=None, attention_mask=lay.pad_masks.long())
    assert px.suffix_position_start(pos, lay.pad_masks) == lay.S == 24


def test_suffix_position_start_matches_official_formula(px):
    """suffix_position_start == _build_full_position_ids' p0 (:741-747), incl. padding."""
    pos = torch.tensor([[[0, 1, 2, 3]], [[0, 1, 2, 3]], [[0, 1, 5, 3]]])   # max = 5
    pad = torch.ones(1, 4, dtype=torch.bool)
    assert px.suffix_position_start(pos, pad) == 6
    pad2 = torch.tensor([[True, True, False, True]])       # mask out the 5 -> max = 3
    assert px.suffix_position_start(pos, pad2) == 4


# ---------------------------------------------------------------------------
# Suffix mask blocking
# ---------------------------------------------------------------------------
def test_suffix_blocked_from_future_depth_tail(px, acfg):
    """block_future_depth_to_action zeroes the TAIL 8 (future_depth) prefix columns."""
    prefix_len = 100
    m = torch.ones(51, prefix_len, dtype=torch.bool)
    px.apply_suffix_prefix_blocking_(m, prefix_len, acfg)
    assert not m[:, prefix_len - 8:].any(), "future_depth queries must be hidden from the suffix"
    assert m[:, :prefix_len - 8].all(), "everything before the tail stays visible"


def test_no_blocking_when_flags_off(px):
    c = px.AlignConfig(block_future_depth_to_action=False, block_suffix_to_future_video=False)
    m = torch.ones(51, 100, dtype=torch.bool)
    px.apply_suffix_prefix_blocking_(m, 100, c)
    assert m.all()


def test_suffix_mask_separates_real_prefix_from_execution_padding(px, acfg):
    tree = ast.parse(RUNTIME_PY.read_text())
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_make_att_2d_masks", "build_suffix_mask_bool"}
    }
    assert set(functions) == {"_make_att_2d_masks", "build_suffix_mask_bool"}
    ns = {
        "torch": torch,
        "N_ACTION": 50,
        "apply_suffix_prefix_blocking_": px.apply_suffix_prefix_blocking_,
    }
    exec(
        compile(
            ast.Module(
                body=[functions["_make_att_2d_masks"],
                      functions["build_suffix_mask_bool"]],
                type_ignores=[],
            ),
            "<lingbot2_suffix_mask>",
            "exec",
        ),
        ns,
    )
    mask = ns["build_suffix_mask_bool"](
        225, acfg, real_prefix_len=219
    )
    assert mask.shape == (51, 276)
    # Future-depth is the final eight REAL rows [211,219); adapter padding is
    # [219,225). Both are invisible, while the preceding REAL row is visible.
    assert mask[:, 210].all()
    assert not mask[:, 211:225].any()
    assert mask[:, 225:].any()


def test_suffix_mask_rejects_real_length_outside_execution(px, acfg):
    tree = ast.parse(RUNTIME_PY.read_text())
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_make_att_2d_masks", "build_suffix_mask_bool"}
    }
    ns = {
        "torch": torch,
        "N_ACTION": 50,
        "apply_suffix_prefix_blocking_": px.apply_suffix_prefix_blocking_,
    }
    exec(compile(ast.Module(body=list(functions.values()), type_ignores=[]),
                 "<lingbot2_suffix_mask>", "exec"), ns)
    with pytest.raises(ValueError, match="real_prefix_len"):
        ns["build_suffix_mask_bool"](225, acfg, real_prefix_len=226)


# ---------------------------------------------------------------------------
# RoPE theta = 5e6 + the M-RoPE -> 1D reduction
# ---------------------------------------------------------------------------
def test_runtime_never_hardcodes_v1_rope_base():
    """runtime.py/convert.py must not carry V1's ROPE_BASE = 10000 constant."""
    for f in (RUNTIME_PY, CONVERT_PY):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and "ROPE_BASE" in t.id.upper():
                        pytest.fail(f"{f.name} defines {t.id} — theta must come from the config")


def test_resolve_rope_theta_reads_config(cfg):
    """resolve_rope_theta pulls 5e6 out of the REAL config and refuses to guess."""
    src = RUNTIME_PY.read_text()
    ns: dict = {}
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "resolve_rope_theta")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<rt>", "exec"), ns)
    resolve = ns["resolve_rope_theta"]

    assert resolve(cfg.text_config) == 5_000_000.0, "expert theta must be the VLM's 5e6, not 10000"

    class Bare:                       # no rope_parameters / rope_scaling / rope_theta at all
        pass
    with pytest.raises(ValueError, match="refusing to fall back to V1's 10000"):
        resolve(Bare())


def test_mrope_all_equal_axes_reduces_to_1d_rope(cfg):
    """REGRESSION: all-3-axes-equal M-RoPE == plain 1-D RoPE at the VLM's theta.

    This is what makes the plain-1D expert exact for the suffix, whose position ids are
    `suffix_1d.unsqueeze(0).expand(3, -1, -1)` (modeling_lingbot_vla_v2.py:746). Previously
    measured max|diff| cos 2.2e-6 / sin 1.3e-6 (fp32 rounding).
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    tc = cfg.text_config
    theta = tc.rope_parameters["rope_theta"]
    hd = tc.head_dim
    assert theta == 5_000_000 and tc.rope_parameters["mrope_interleaved"] is True
    assert tc.rope_parameters["mrope_section"] == [24, 20, 20]

    rot = Qwen3VLTextRotaryEmbedding(tc)
    pos1d = torch.arange(42, 42 + 51).view(1, -1)          # a realistic suffix span at p0=42
    pos3 = pos1d.unsqueeze(0).expand(3, -1, -1).contiguous()
    cos, sin = rot(torch.zeros(1, pos1d.shape[1], 1, hd), pos3)

    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    fr = pos1d[0].double()[:, None] * inv[None, :]
    cos1d = torch.cat([fr.cos(), fr.cos()], dim=-1).float()
    sin1d = torch.cat([fr.sin(), fr.sin()], dim=-1).float()

    assert (cos.reshape(-1, hd) - cos1d).abs().max().item() < 1e-5
    assert (sin.reshape(-1, hd) - sin1d).abs().max().item() < 1e-5


def test_wrong_theta_would_be_caught(cfg):
    """Sanity: the 1-D check above is not vacuous — theta=10000 must NOT match."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    tc = cfg.text_config
    hd = tc.head_dim
    rot = Qwen3VLTextRotaryEmbedding(tc)
    pos1d = torch.arange(42, 42 + 51).view(1, -1)
    pos3 = pos1d.unsqueeze(0).expand(3, -1, -1).contiguous()
    cos, _ = rot(torch.zeros(1, pos1d.shape[1], 1, hd), pos3)
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    fr = pos1d[0].double()[:, None] * inv[None, :]
    cos_wrong = torch.cat([fr.cos(), fr.cos()], dim=-1).float()
    assert (cos.reshape(-1, hd) - cos_wrong).abs().max().item() > 0.1


# ---------------------------------------------------------------------------
# Direct Vision -> persistent prefix fixed-DMA contract (pure CPU / AST)
# ---------------------------------------------------------------------------
def _exec_runtime_functions(*names):
    tree = ast.parse(RUNTIME_PY.read_text())
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in wanted} == set(names)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "<rt>", "exec"), ns)
    return [ns[name] for name in names]


def test_prompt_bucket_refresh_keeps_every_graph_baked_address_stable():
    scatter, refresh = _exec_runtime_functions(
        "_scatter_rows_", "_refresh_prefix_memo_"
    )
    assert callable(scatter)
    memo = {
        "key": "A",
        "embeds": torch.zeros(1, 6, 2),
        "dense": [torch.zeros(6, 2) for _ in range(3)],
        "position_ids": torch.zeros(3, 1, 6, dtype=torch.int32),
        # Decoder-native stable owner: [S, 3], not a per-call contiguous temp.
        "position_ids_rpu": torch.zeros(6, 3, dtype=torch.int32),
        "rope_cos_il": torch.zeros(6, 2, dtype=torch.float16),
        "rope_sin_il": torch.zeros(6, 2, dtype=torch.float16),
        "S": 6,
        "p0": 4,
        "runs": [(1, 2)],
        "offs": [0],
    }
    owners = [memo["embeds"], *memo["dense"], memo["position_ids"],
              memo["position_ids_rpu"], memo["rope_cos_il"],
              memo["rope_sin_il"]]
    assert not any(torch.is_inference(owner) for owner in owners)
    pointers = tuple(owner.data_ptr() for owner in owners)
    position_epoch = memo["position_ids_rpu"]._version
    cos_epoch = memo["rope_cos_il"]._version
    sin_epoch = memo["rope_sin_il"]._version

    def apply(tag, value, run):
        embeds = torch.full((1, 6, 2), value)
        positions = torch.full((3, 1, 6), value, dtype=torch.int32)
        rope_cos = torch.full((6, 2), value, dtype=torch.float16)
        rope_sin = torch.full((6, 2), -value, dtype=torch.float16)
        features = [torch.full((2, 2), value + tap) for tap in range(3)]
        with torch.inference_mode():
            refresh(
                memo, memo_key=tag, embeds=embeds, position_ids=positions,
                rope_cos_il=rope_cos, rope_sin_il=rope_sin,
                deepstack_features=features, runs=[run], offs=[0], p0=value,
            )

    apply("B", 7, (3, 2))
    assert tuple(owner.data_ptr() for owner in owners) == pointers
    assert memo["position_ids_rpu"]._version == position_epoch + 1
    assert memo["rope_cos_il"]._version == cos_epoch + 1
    assert memo["rope_sin_il"]._version == sin_epoch + 1
    assert memo["embeds"].eq(7).all()
    assert memo["position_ids"].eq(7).all()
    assert memo["position_ids_rpu"].eq(7).all()
    assert memo["rope_cos_il"].eq(7).all()
    assert memo["rope_sin_il"].eq(-7).all()
    assert not memo["dense"][0][:3].any()
    assert memo["dense"][0][3:5].eq(7).all()

    apply("A", 4, (1, 2))
    assert tuple(owner.data_ptr() for owner in owners) == pointers
    assert memo["position_ids_rpu"]._version == position_epoch + 2
    assert memo["rope_cos_il"]._version == cos_epoch + 2
    assert memo["rope_sin_il"]._version == sin_epoch + 2
    assert memo["key"] == "A" and memo["p0"] == 4
    assert memo["dense"][0][1:3].eq(4).all()
    assert not memo["dense"][0][3:].any()


def test_exact_three_camera_zero_copy_profile_contract(px, cfg, acfg, ids):
    """Freeze the LingBot2 route geometry before any physical-fit admission."""
    vc, tc = cfg.vision_config, cfg.text_config
    assert (vc.depth, vc.hidden_size, vc.intermediate_size,
            vc.num_heads, vc.hidden_size // vc.num_heads,
            vc.out_hidden_size) == (24, 1024, 4096, 16, 64, 2560)
    assert list(vc.deepstack_visual_indexes) == [5, 11, 17]
    assert vc.spatial_merge_size == 2
    assert (tc.num_hidden_layers, tc.hidden_size, tc.intermediate_size,
            tc.num_attention_heads, tc.num_key_value_heads,
            tc.head_dim) == (36, 2560, 9728, 32, 8, 128)

    grid = torch.tensor([[1, 16, 16]] * 3)
    patches_per_image = int(grid[0].prod().item())
    merged_per_image = ((int(grid[0, 1]) // vc.spatial_merge_size) *
                        (int(grid[0, 2]) // vc.spatial_merge_size))
    assert grid.shape[0] * patches_per_image == 768
    assert grid.shape[0] * merged_per_image == 192

    lang = torch.arange(1000, 1011, dtype=torch.long).view(1, -1)
    layout = px.build_prefix_layout(
        n_images=3, num_patch=merged_per_image, lang_ids=lang,
        cfg=acfg, ids=ids)
    contiguous_runs, = _exec_runtime_functions("_contiguous_runs")
    runs = contiguous_runs(layout.visual_pos_masks.squeeze(0))
    assert layout.S == 225
    assert layout.spans["language"] == (198, 209)
    assert runs == [(1, 64), (67, 64), (133, 64)]

    tree = ast.parse(RUNTIME_PY.read_text())
    builder = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "_build_lingbot_vla_v2_impl")
    target_assignment = next(
        n for n in ast.walk(builder)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "deepstack_text_layers"
                for t in n.targets))
    assert ast.unparse(target_assignment.value) == (
        "list(range(len(cfg.vision_config.deepstack_visual_indexes)))")
    deepstack_text_layers = list(range(len(vc.deepstack_visual_indexes)))
    assert deepstack_text_layers == [0, 1, 2]

    install = next(
        n for n in ast.walk(builder)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func).endswith("install_qwen3_vl_text_for_rpu"))
    keywords = {kw.arg: ast.unparse(kw.value) for kw in install.keywords}
    assert keywords["deepstack_lang_layers"] == "deepstack_text_layers"
    assert keywords["enable_deepstack"] == "True"


def _symbolic_row_partition(total_rows: int, chunk_cap: int):
    """Pure row partition used only to prove chunk-independent route coverage.

    This is intentionally not the native Text chunk resolver: resolving the
    production chunk requires the prepared model layout.  Therefore callers
    must not treat any partition produced here as physical-fit evidence.
    """
    assert total_rows > 0 and chunk_cap > 0
    return [(begin, min(chunk_cap, total_rows - begin))
            for begin in range(0, total_rows, chunk_cap)]


@pytest.mark.parametrize("symbolic_text_chunk_cap", [64, 96, 128, 225])
def test_exact_three_camera_sequential_n256_phase_schedule_contract(
        px, cfg, acfg, ids, symbolic_text_chunk_cap):
    """Prove the slice schedule for sequential N256 without admitting SPM."""
    vc, tc = cfg.vision_config, cfg.text_config
    assert (vc.depth, vc.hidden_size, vc.intermediate_size,
            vc.num_heads, vc.hidden_size // vc.num_heads,
            vc.out_hidden_size) == (24, 1024, 4096, 16, 64, 2560)
    deepstack_pairs = list(zip(vc.deepstack_visual_indexes,
                               range(len(vc.deepstack_visual_indexes))))
    assert deepstack_pairs == [(5, 0), (11, 1), (17, 2)]
    assert (tc.num_hidden_layers, tc.hidden_size, tc.intermediate_size,
            tc.num_attention_heads, tc.num_key_value_heads,
            tc.head_dim) == (36, 2560, 9728, 32, 8, 128)

    grids = torch.tensor([[1, 16, 16]] * 3)
    patches_per_image = [int(row.prod().item()) for row in grids]
    merged_per_image = [
        (int(row[1]) // vc.spatial_merge_size) *
        (int(row[2]) // vc.spatial_merge_size)
        for row in grids
    ]
    assert patches_per_image == [256, 256, 256]
    assert merged_per_image == [64, 64, 64]

    lang = torch.arange(1000, 1011, dtype=torch.long).view(1, -1)
    layout = px.build_prefix_layout(3, 64, lang, acfg, ids)
    contiguous_runs, = _exec_runtime_functions("_contiguous_runs")
    visual_runs = contiguous_runs(layout.visual_pos_masks.squeeze(0))
    assert layout.S == 225
    assert visual_runs == [(1, 64), (67, 64), (133, 64)]

    # Each inclusive [event,event] Vision phase owns the same logical scratch
    # slot, so the three N256 executions are sequential rather than one N768
    # scratch allocation.  No physical address or byte extent is invented.
    vision_phases = [
        {"event": image, "image": image, "patches": 256,
         "scratch_slot": "vision_n256"}
        for image in range(3)
    ]
    assert [phase["event"] for phase in vision_phases] == [0, 1, 2]
    assert len({phase["scratch_slot"] for phase in vision_phases}) == 1
    assert all(phase["patches"] == 256 for phase in vision_phases)

    # Retain one compact slice per (tap,image), then release the three slices
    # for a tap at its Text consumer layer.  This deliberately never constructs
    # a long-lived [192,2560] DeepStack dense port.  The nine slice lifetimes may
    # still overlap; this topology-only test does not calculate or admit a peak.
    text_event_base = len(vision_phases)
    retained_slices = [
        {"tap": tap, "text_layer": text_layer, "image": image,
         "shape": (64, tc.hidden_size), "produced_event": image,
         "last_read_event": text_event_base + text_layer,
         "destination_run": visual_runs[image]}
        for tap, text_layer in deepstack_pairs
        for image in range(3)
    ]
    assert len(retained_slices) == 9
    assert all(item["shape"] == (64, 2560) for item in retained_slices)
    assert (192, 2560) not in {item["shape"] for item in retained_slices}
    assert all(item["produced_event"] <= item["last_read_event"]
               for item in retained_slices)

    # The actual native Text chunk is deliberately unknown here.  For several
    # symbolic caps, partition S=225 with a pure function and prove that the
    # chunk/run intersections cover every visual destination row and every
    # source row of all three 64-row slices at each Text layer, including the
    # final visual row 196.
    text_chunks = _symbolic_row_partition(layout.S,
                                          symbolic_text_chunk_cap)
    assert text_chunks[0][0] == 0
    assert sum(count for _begin, count in text_chunks) == layout.S
    assert text_chunks[-1][0] + text_chunks[-1][1] == layout.S

    expected_destination_rows = [
        row for begin, count in visual_runs
        for row in range(begin, begin + count)
    ]
    for _tap, text_layer in deepstack_pairs:
        consumed_destination_rows = []
        consumed_source_rows = {image: [] for image in range(3)}
        for chunk_begin, chunk_count in text_chunks:
            chunk_end = chunk_begin + chunk_count
            for image, (run_begin, run_count) in enumerate(visual_runs):
                overlap_begin = max(chunk_begin, run_begin)
                overlap_end = min(chunk_end, run_begin + run_count)
                if overlap_begin >= overlap_end:
                    continue
                consumed_destination_rows.extend(
                    range(overlap_begin, overlap_end))
                consumed_source_rows[image].extend(
                    range(overlap_begin - run_begin,
                          overlap_end - run_begin))

        assert consumed_destination_rows == expected_destination_rows
        assert consumed_destination_rows[-1] == 196
        assert all(rows == list(range(64))
                   for rows in consumed_source_rows.values())
        assert {item["destination_run"] for item in retained_slices
                if item["text_layer"] == text_layer} == set(visual_runs)


def test_direct_prefix_group_accepts_report_geometry_only_as_one_group():
    validate, = _exec_runtime_functions("_validate_direct_prefix_group")
    grid = torch.tensor([[1, 16, 16]] * 3)
    validate(grid, patches_per_image=256, batch_cap=3, cache_capacity=768)

    # Same patch count is insufficient: exact grid identity drives RoPE/window layout.
    mixed = torch.tensor([[1, 16, 16], [1, 8, 32], [1, 16, 16]])
    with pytest.raises(ValueError, match="exactly the same"):
        validate(mixed, patches_per_image=256, batch_cap=3, cache_capacity=768)
    with pytest.raises(ValueError, match="batch cap"):
        validate(grid, patches_per_image=256, batch_cap=2, cache_capacity=768)
    with pytest.raises(ValueError, match="KV capacity"):
        validate(grid, patches_per_image=256, batch_cap=3, cache_capacity=767)
    # The generic planner does not invoke minibatch SDPA's predicate for g=1.
    validate(torch.tensor([[1, 8, 8]]), patches_per_image=64,
             batch_cap=1, cache_capacity=64)


def test_direct_prefix_group_rejects_generic_adapter_split_point():
    validate, = _exec_runtime_functions("_validate_direct_prefix_group")
    # Four 128-row blocks/image: g=2 gives grid_dim_x=8, but g=3 gives 12,
    # so qwen3_vl.vision stops at two images and emits a second group.
    with pytest.raises(ValueError, match="intermediate group size 3"):
        validate(torch.tensor([[1, 16, 32]] * 3), patches_per_image=512,
                 batch_cap=3, cache_capacity=1536)


def test_direct_prefix_run_validator_guards_fixed_dma_bounds():
    validate, = _exec_runtime_functions("_validate_prefix_scatter_runs")
    validate([(1, 64), (67, 64), (133, 64)], prefix_rows=220,
             expected_rows=192, expected_runs=3)
    with pytest.raises(ValueError, match="invalid visual run"):
        validate([(1, 64), (60, 64), (133, 64)], prefix_rows=220,
                 expected_rows=192, expected_runs=3)
    with pytest.raises(ValueError, match="cover"):
        validate([(1, 64), (67, 64), (133, 63)], prefix_rows=220,
                 expected_rows=192, expected_runs=3)


def test_direct_prefix_rebind_refuses_ready_before_mutating_any_cache():
    clear_for_rebind, = _exec_runtime_functions("_clear_graph_caches_for_rebind")

    class Cache:
        def __init__(self, frozen):
            self.frozen = frozen
            self.events = []

        def is_frozen(self):
            return self.frozen

        def clear(self):
            self.events.append("clear")

    vision, vlm = Cache(False), Cache(True)
    with pytest.raises(RuntimeError, match="READY"):
        clear_for_rebind(((vision, "Vision"), (vlm, "VLM")))
    assert vision.events == [] and vlm.events == []

    vision, vlm = Cache(False), Cache(False)
    clear_for_rebind(((vision, "Vision"), (vlm, "VLM")))
    assert vision.events == ["clear"] and vlm.events == ["clear"]


def test_direct_prefix_publishes_only_after_setter_and_before_vision_build():
    tree = ast.parse(RUNTIME_PY.read_text())
    policy = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "LingbotVlaV2Policy")
    method = next(n for n in policy.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_build_prefix_direct")
    memo_publish = min(
        n.lineno for n in ast.walk(method)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "self._pfx_memo" for t in n.targets)
    )
    setter = min(n.lineno for n in _find_call(
        method, "qwen3vl_vision_set_prefix_scatter"))
    vision = min(
        n.lineno for n in ast.walk(method)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "self._visual"
    )
    assert setter < memo_publish < vision


def test_private_no_output_path_cannot_change_public_qwen_vision_semantics():
    tree = ast.parse(QWEN3VL_VISION_PY.read_text())
    forward = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "_rpu_vision_forward")
    source = ast.unparse(forward)
    assert "_rpu_prefix_scatter_consume_only_allowed" in source
    assert "len(groups) != 1" in source
    assert "return BaseModelOutputWithDeepstackFeatures" in source

    parents = {}
    for parent in ast.walk(forward):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    pop = next(_find_call(forward, "qwen3vl_vision_pop_merged"))
    guarded = False
    node = pop
    while node in parents:
        node = parents[node]
        if (isinstance(node, ast.If)
                and "not _prefix_scatter_consume_only" in ast.unparse(node.test)):
            guarded = True
            break
    assert guarded, "generic pop/clone must be unreachable in private consume-only mode"


# ---------------------------------------------------------------------------
# GraphCache signature / cos_sin_offset wiring (static AST checks — no device)
# ---------------------------------------------------------------------------
def _find_call(tree, attr_chain):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if ast.unparse(node.func).endswith(attr_chain):
                yield node


def test_denoise_graph_signature_separates_dynamic_ready_rope_from_legacy_identity():
    """Legacy keys S/REAL/p0; READY keys S after exact stable-table refresh."""
    tree = ast.parse(RUNTIME_PY.read_text())
    sigs = [n for n in _find_call(tree, "GraphSignature")]
    denoise = None
    for n in sigs:
        kw = {k.arg: k.value for k in n.keywords}
        if isinstance(kw.get("op_id"), ast.Constant) and kw["op_id"].value == "lingbot2_expert_denoise":
            denoise = kw
    assert denoise is not None, "lingbot2_expert_denoise GraphSignature not found"
    assert ast.unparse(denoise["dyn_dims"]) == (
        "[*denoise_dyn_dims, *action_plan.graph_key_words()]"
    )
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "denoise_dyn_dims" for target in node.targets)
    )
    value = ast.unparse(assignment.value)
    assert "[EXP_LAYERS, S, PREFIX_POLICY_VERSION]" in value
    assert "[EXP_LAYERS, S, real_S, p0, PREFIX_POLICY_VERSION]" in value
    refresh = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_ready_suffix_rope"
    )
    refresh_source = ast.unparse(refresh)
    assert "build_suffix_rope_window_cpu" in refresh_source
    assert "return 0" in refresh_source


def test_expert_forward_passes_S_and_p0():
    """_exp_run passes position=S, RoPE=p0, and the immutable descriptor."""
    tree = ast.parse(RUNTIME_PY.read_text())
    calls = [ast.unparse(n) for n in _find_call(tree, "lingbot_v2_moe_forward")]
    assert calls, "no lingbot_v2_moe_forward call found"
    assert any(c.endswith("mask_r, S, p0, -1, descriptor)") for c in calls), \
        f"expected a (..., mask_r, S, p0, -1, descriptor) call; got {calls}"
    assert any(
        c.endswith("mask_r, S, p0, int(adarms_step), descriptor)")
        for c in calls
    ), "expected direct-schedule forward to carry (..., S, p0, step, descriptor)"
    assert not any(c.endswith("mask_r, S)") for c in calls)


def test_incomplete_action_abi_is_rejected_during_construction():
    """Production has no old-.so action fallback without a descriptor."""
    src = RUNTIME_PY.read_text()
    assert "_has_cos_sin_offset" in src
    assert "if not self._has_cos_sin_offset:" in src
    assert "complete planned_stage_descriptor" in src
    assert "if p0 != S and not self._has_cos_sin_offset:" not in src


def test_python_call_matches_cpp_registered_schema():
    """Cross-agent contract: our positional call must match the C++ `m.def` arg order EXACTLY.

    The Python and C++ halves are owned by different agents, so a silent signature drift is a
    real risk. This parses the registered schema out of main's dispatch-registration include and
    pins the argument order our `_exp_run` relies on. It needs no .so and no device.
    """
    assert RPU_REGISTRATIONS.exists(), f"missing {RPU_REGISTRATIONS}"
    src = RPU_REGISTRATIONS.read_text()
    i = src.find('m.def("lingbot_v2_moe_forward(')
    assert i != -1, "lingbot_v2_moe_forward not registered in this tree"
    # stitch the adjacent C string literals of the m.def(...) back together
    chunk = src[i:src.index(');', i)]
    schema = "".join(p.split('"')[1] for p in chunk.splitlines() if '"' in p)

    args = schema[schema.index("(") + 1: schema.rindex(")")].split(", ")
    names = [a.split()[-1].split("=")[0] for a in args]
    assert names == ["handle", "hidden_states", "k_caches", "v_caches",
                     "attention_mask", "position", "cos_sin_offset",
                     "adarms_schedule_step", "planned_stage_descriptor"], \
        f"C++ arg order changed -> our positional call is now wrong: {names}"
    assert "int cos_sin_offset=-1" in schema, \
        "cos_sin_offset must default to -1 (back-compat with a single `position`)"
    assert "int adarms_schedule_step=-1" in schema, \
        "schedule step must default to -1 so existing 6/7-argument calls stay compatible"
    # ...and that is exactly what _exp_run passes, positionally, in this order.
    rt = RUNTIME_PY.read_text()
    assert "mask_r, S, p0, -1, descriptor)" in rt, \
        "runtime must pass position=S, cos_sin_offset=p0, and the descriptor"


def test_adarms_direct_schedule_is_one_bind_and_mutually_exclusive_per_step():
    tree = ast.parse(RUNTIME_PY.read_text())
    policy = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "LingbotVlaV2Policy")
    init = next(n for n in policy.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    action = next(n for n in policy.body
                  if isinstance(n, ast.FunctionDef) and n.name == "get_action")

    fold_line = min(
        n.lineno for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "self._fold_all" for t in n.targets))
    bind_line = min(
        n.lineno for n in _find_call(init, "lingbot_v2_moe_bind_adarms_schedule"))
    assert fold_line < bind_line, "full schedule must exist before its one-time native bind"

    parents = {}
    for parent in ast.walk(action):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    setter = next(_find_call(action, "lingbot_v2_moe_set_adarms_step_mutable"))
    node = setter
    while node in parents and not isinstance(node, ast.If):
        node = parents[node]
    assert isinstance(node, ast.If)
    assert ast.unparse(node.test) == "not self._adarms_direct_schedule"
    assert "adarms_step=k if self._adarms_direct_schedule else None" in ast.unparse(action)


def test_adarms_direct_schedule_preflight_precedes_rpu_materialization():
    tree = ast.parse(RUNTIME_PY.read_text())
    impl = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "_build_lingbot_vla_v2_impl")
    validate_line = min(n.lineno for n in _find_call(
        impl, "_validate_adarms_direct_schedule"))
    materialize_line = min(n.lineno for n in _find_call(impl, "torch.rpu.set_ddr_flush"))
    assert validate_line < materialize_line
