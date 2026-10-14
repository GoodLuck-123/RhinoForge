#!/usr/bin/env python3
"""scripts/check_import_graph_v5.py — v5 DAG enforcer (full mode)."""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys
import tempfile
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RPU_PKG = REPO_ROOT / "python" / "rpu_backend"
API_ROOT = RPU_PKG / "api"
RUNTIME_ROOT = RPU_PKG / "runtime"
ADAPTERS_ROOT = RPU_PKG / "adapters"


# --- api/ scope (DAG-04 carryover from v5-01a-1) --------------------------

API_FORBIDDEN_TOPLEVEL_PREFIXES = (
    "rpu_backend.transformers",
    "rpu_backend.adapters",
)


# --- runtime/ scope (DAG-01/02/03) ----------------------------------------

# DAG-01 + DAG-02: runtime ↛ adapters / api
RUNTIME_FORBIDDEN_PREFIXES = (
    "rpu_backend.adapters",
    "rpu_backend.api",
)

# DAG-03: model-specific arch tokens forbidden in runtime/.
RUNTIME_ARCH_TOKENS = (
    "qwen3", "llama", "pi05", "gemma", "adarms", "siglip", "qwenpi05",
)

# HND-05 + DAG-03 whitelist: per-runtime-file exemptions for arch tokens.
RUNTIME_ARCH_WHITELIST = {
    # hw_attrs.py: _rpu_*_handle attribute names + per-arch docstring mentions
    # are intentional (the hardware whitelist is the authoritative source).
    "hw_attrs.py": {"qwen3", "llama", "pi05", "gemma", "adarms", "siglip", "qwenpi05"},
    # decoder.py: function-name mentions in patched-class docstrings + comments
    # ("Qwen3RMSNorm", "Qwen3RotaryEmbedding") are doc-only, not arch coupling.
    "decoder.py": {"qwen3", "llama", "pi05", "gemma", "adarms", "siglip"},
}


# --- adapters/ scope (DAG-05) ---------------------------------------------

# Pi05 internal package siblings (single-arch subpackage allowed to cross-import).
PI05_SIBLINGS = {
    "__init__.py", "gemma.py", "adarms.py", "siglip.py",
    "loader.py", "weights.py", "runtime.py", "patches.py",
}

# Shared peer modules that policy adapters legitimately reuse:
#   * SigLIP — shared ViT direct-handle path (pi05 / paligemma / halo). HALO's vision
#     encoder IS SigLIP-SO400M-navit; vit_cache reuses the siglip weight pad/swizzle
#     converters (_convert_siglip_weights_for_rpu / _siglip_weight_to_rpu) to guarantee
#     byte-identical conversion — the same shared-ViT pattern as pi05 / paligemma.
#   * qwen3_vl — GR00T-N1.7 and LingBot-VLA-V2 embed Qwen3-VL backbones and reuse
#     the qwen3_vl vision + text install path (same shared-backbone pattern as siglip).
#   * qwen3_5 — G0.5 embeds a Qwen3.5 VLM and action expert and reuses that exact
#     text/vision install and execution path.
#   * wall_oss — InternVLA-N1's Qwen2.5-VL System-2 reuses the same causal-decoder
#     streaming checkpoint and weight-install helpers; moving that loader would duplicate
#     its non-mmap memory contract.
#   * internvla_n1._policy — NavDP is InternVLA-N1's System-1 head and shares only
#     its fail-closed opt-in and asset-manifest policy.  Other InternVLA modules and
#     the reverse edge stay forbidden.
#   * Qwen3.5 dense/MoE — the MoE adapter reuses the same Vision tower, while
#     Vision attests the MoE native topology and exact parent-close callback.
#     These four module targets are an indivisible, narrowly scoped shared-Vision
#     contract; other dense/MoE peer modules remain forbidden.
# Keep this narrow: any OTHER cross-arch import stays forbidden by DAG-05 — e.g. the
# shared partial-mRoPE host helper was moved to runtime/ (rpu_backend.runtime.
# rope_partial) instead of allowlisting a gr00t<->wall_oss sibling edge. (SigLIP
# converters cannot move to runtime/ — they carry the `siglip` arch token forbidden
# there by DAG-03; the allowlist is the intended mechanism for shared-ViT reuse.)
ALLOWED_SHARED_PEER_IMPORTS = {
    ("pi05", "siglip"),
    ("paligemma", "siglip"),
    ("halo", "siglip"),
    ("gr00t", "qwen3_vl"),
    ("lingbot_vla_v2", "qwen3_vl"),
    ("g05", "qwen3_5"),
    ("internvla_n1", "wall_oss"),
}

# Exceptions narrower than an architecture pair.  Keep the complete import
# target here so adding one shared policy cannot silently expose the peer's
# backbone/runtime modules to the importing architecture.
ALLOWED_EXACT_PEER_IMPORTS = {
    # RhinoVLA v3 embeds this exact Qwen3-VL Vision tower and reuses its cold
    # W8 quantize/swizzle converters; text/runtime and reverse imports stay closed.
    ("rhinovla", "rpu_backend.adapters.qwen3_vl.vision"),
    ("navdp", "rpu_backend.adapters.internvla_n1._policy"),
    ("qwen3_5", "rpu_backend.adapters.qwen3_5_moe"),
    ("qwen3_5", "rpu_backend.adapters.qwen3_5_moe.cores"),
    ("qwen3_5_moe", "rpu_backend.adapters.qwen3_5.cores"),
    ("qwen3_5_moe", "rpu_backend.adapters.qwen3_5.vision"),
}


# --- forbidden HND-05 attribute literals ---------------------------------

HND05_LEGACY_LITERALS = {
    "_rpu_qwen3_handle",
    "_rpu_gemma_handle",
    "_rpu_adarms_handle",
    "_rpu_siglip_handle",
}


# ============================================================================
# Per-scope checker classes
# ============================================================================


class ApiScopeChecker(ast.NodeVisitor):
    """api/ scope — top-level forbidden imports of transformers/* + adapters/*."""

    def __init__(self, relpath: pathlib.Path):
        self.relpath = relpath
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self._depth_into_callable = 0
        self._depth_into_type_checking = 0

    def visit_FunctionDef(self, node):
        self._depth_into_callable += 1
        self.generic_visit(node)
        self._depth_into_callable -= 1

    def visit_AsyncFunctionDef(self, node):
        self._depth_into_callable += 1
        self.generic_visit(node)
        self._depth_into_callable -= 1

    def visit_ClassDef(self, node):
        # Class body executes at import time — keep depth same as module.
        self.generic_visit(node)

    def visit_If(self, node):
        test = node.test
        is_tc = (
            (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
        )
        if is_tc:
            self._depth_into_type_checking += 1
            self.generic_visit(node)
            self._depth_into_type_checking -= 1
        else:
            self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self._record(alias.name, node.lineno)

    def visit_ImportFrom(self, node):
        if node.module is None:
            return
        if node.module == "rpu_backend":
            for alias in node.names:
                self._record(f"rpu_backend.{alias.name}", node.lineno)
            return
        self._record(node.module, node.lineno)

    def _record(self, modname: str, lineno: int) -> None:
        if not any(modname.startswith(p) for p in API_FORBIDDEN_TOPLEVEL_PREFIXES):
            return
        if self._depth_into_callable > 0:
            return
        if self._depth_into_type_checking > 0:
            self.warnings.append(
                f"{self.relpath}:{lineno}: TYPE_CHECKING-only import of `{modname}`"
            )
            return
        self.failures.append(
            f"{self.relpath}:{lineno}: TOP-LEVEL import of `{modname}` from "
            f"under `api/` violates ADR §3.2 DAG (api ↛ transformers/adapters)"
        )


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Return ids() of every Constant node that is a docstring (first statement
    of a Module / FunctionDef / AsyncFunctionDef / ClassDef body that is a string
    Expression). Docstrings are documentation prose — exempted from arch-token scans.
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, 'body', None) or []
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            ids.add(id(body[0].value))
    return ids


class RuntimeScopeChecker(ast.NodeVisitor):
    """runtime/ scope — forbidden imports of adapters/api + model-specific tokens
    in imports, string literals, identifiers, names.

    Body-local imports (inside FunctionDef / AsyncFunctionDef) are exempted from
    DAG-01/02 (lazy imports OK). Class bodies execute at import time, so they ARE
    scanned. Docstrings are exempted from DAG-03 arch-token scans.
    """

    def __init__(self, relpath: pathlib.Path, docstring_ids: set[int]):
        self.relpath = relpath
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self._depth_into_callable = 0
        self._depth_into_type_checking = 0
        self._docstring_ids = docstring_ids
        # Per-file arch token whitelist (filename basename).
        self._whitelisted_tokens = RUNTIME_ARCH_WHITELIST.get(relpath.name, set())

    def _arch_hits(self, text: str) -> set[str]:
        """Return arch tokens contained in `text` (case-insensitive substring match),
        excluding those in this file's whitelist."""
        lo = text.lower()
        return {tok for tok in RUNTIME_ARCH_TOKENS
                if tok in lo and tok not in self._whitelisted_tokens}

    def visit_FunctionDef(self, node):
        hits = self._arch_hits(node.name)
        if hits:
            self.failures.append(
                f"{self.relpath}:{node.lineno}: runtime/ function name `{node.name}` "
                f"contains arch token(s) {sorted(hits)} (DAG-03)"
            )
        self._depth_into_callable += 1
        self.generic_visit(node)
        self._depth_into_callable -= 1

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        hits = self._arch_hits(node.name)
        if hits:
            self.failures.append(
                f"{self.relpath}:{node.lineno}: runtime/ class name `{node.name}` "
                f"contains arch token(s) {sorted(hits)} (DAG-03)"
            )
        # Class body executes at import time → keep callable depth same as outer.
        self.generic_visit(node)

    def visit_If(self, node):
        test = node.test
        is_tc = (
            (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
        )
        if is_tc:
            self._depth_into_type_checking += 1
            self.generic_visit(node)
            self._depth_into_type_checking -= 1
        else:
            self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self._check_import_target(alias.name, node.lineno)

    def visit_ImportFrom(self, node):
        if node.module is None:
            return
        if node.module == "rpu_backend":
            for alias in node.names:
                self._check_import_target(f"rpu_backend.{alias.name}", node.lineno)
            return
        self._check_import_target(node.module, node.lineno)

    def _check_import_target(self, modname: str, lineno: int) -> None:
        # DAG-01 / DAG-02 — body-local imports are OK (lazy pattern).
        for forbidden in RUNTIME_FORBIDDEN_PREFIXES:
            if modname.startswith(forbidden):
                if self._depth_into_callable > 0:
                    return  # body-local lazy import — allowed
                if self._depth_into_type_checking > 0:
                    self.warnings.append(
                        f"{self.relpath}:{lineno}: TYPE_CHECKING-only import of `{modname}`"
                    )
                    return
                self.failures.append(
                    f"{self.relpath}:{lineno}: runtime/ MUST NOT import `{modname}` "
                    f"(DAG-01/02 — runtime ↛ adapters / runtime ↛ api)"
                )
                return
        # DAG-03: arch tokens in import target.
        hits = self._arch_hits(modname)
        if hits:
            self.failures.append(
                f"{self.relpath}:{lineno}: runtime/ import target `{modname}` "
                f"contains model-specific token(s) {sorted(hits)} (DAG-03)"
            )

    def visit_Constant(self, node):
        # DAG-03 + HND-05: string-literal scan.
        if not isinstance(node.value, str):
            return
        # Docstrings are documentation; skip arch-token + HND-05 scans.
        if id(node) in self._docstring_ids:
            return
        # HND-05: forbid legacy per-arch handle attribute names everywhere.
        if node.value in HND05_LEGACY_LITERALS:
            self.failures.append(
                f"{self.relpath}:{getattr(node, 'lineno', '?')}: HND-05 forbidden "
                f"legacy literal `{node.value}` (use neutral name)"
            )
            return
        # DAG-03: only flag SHORT string literals (no whitespace) — the realistic
        # surface for arch coupling is identifier-like strings (`qwen3_op`,
        # `LlamaForCausalLM`), not free-form prose.
        if any(ws in node.value for ws in (" ", "\n", "\t")):
            return
        hits = self._arch_hits(node.value)
        if hits:
            self.failures.append(
                f"{self.relpath}:{getattr(node, 'lineno', '?')}: runtime/ string "
                f"literal `{node.value!r}` contains arch token(s) {sorted(hits)} (DAG-03)"
            )

    def visit_JoinedStr(self, node):
        # f-string: recurse into Constant values.
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                self.visit_Constant(v)
            elif isinstance(v, ast.FormattedValue):
                self.generic_visit(v)

    def visit_Name(self, node):
        # DAG-03 identifier scan (e.g., `_PI05_MAGIC = 42`).
        hits = self._arch_hits(node.id)
        if hits:
            self.failures.append(
                f"{self.relpath}:{getattr(node, 'lineno', '?')}: runtime/ identifier "
                f"`{node.id}` contains arch token(s) {sorted(hits)} (DAG-03)"
            )

    def visit_arg(self, node):
        hits = self._arch_hits(node.arg)
        if hits:
            self.failures.append(
                f"{self.relpath}:{getattr(node, 'lineno', '?')}: runtime/ arg name "
                f"`{node.arg}` contains arch token(s) {sorted(hits)} (DAG-03)"
            )


class AdapterCrossArchChecker(ast.NodeVisitor):
    """adapters/<arch_b>.py or adapters/<arch_b>/__init__.py MUST NOT import
    `from rpu_backend.adapters.<arch_a>` for arch_a != arch_b. Pi05 internal
    siblings exempted.
    """

    def __init__(self, relpath: pathlib.Path, file_arch: str | None):
        self.relpath = relpath
        # The arch this file belongs to: None means "no arch coupling expected"
        # (e.g., adapters/__init__.py).
        self.file_arch = file_arch
        self.failures: list[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            self._check_target(alias.name, node.lineno)

    def visit_ImportFrom(self, node):
        if node.module is None:
            return
        if node.module == "rpu_backend":
            for alias in node.names:
                self._check_target(f"rpu_backend.{alias.name}", node.lineno)
            return
        self._check_target(node.module, node.lineno)

    def _check_target(self, modname: str, lineno: int) -> None:
        prefix = "rpu_backend.adapters."
        if not modname.startswith(prefix):
            return
        rest = modname[len(prefix):]
        # `rest` is e.g. "qwen3", "llama", "pi05.gemma", ...
        target_arch = rest.split(".", 1)[0]
        if self.file_arch is None:
            return  # parent __init__ etc.: cross-arch imports are fine
        if target_arch == self.file_arch:
            return  # same-arch import: OK
        if (self.file_arch, modname) in ALLOWED_EXACT_PEER_IMPORTS:
            return
        if (self.file_arch, target_arch) in ALLOWED_SHARED_PEER_IMPORTS:
            return
        # Pi05 internal-package siblings exempt: adapters/pi05/<sibling>.py
        # importing from adapters/pi05 is same-arch (handled above); importing
        # adapters/<other> is the violation we catch here.
        self.failures.append(
            f"{self.relpath}:{lineno}: cross-arch import `{modname}` "
            f"(file arch={self.file_arch}, target arch={target_arch}) violates DAG-05"
        )


# ============================================================================
# Tree scanners
# ============================================================================


def _scan_with(checker_cls, paths, *args):
    failures: list[str] = []
    warnings: list[str] = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f"{path}:{e.lineno}: SyntaxError: {e.msg}")
            continue
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        checker = checker_cls(rel, *args)
        checker.visit(tree)
        failures.extend(checker.failures)
        warnings.extend(getattr(checker, "warnings", []))
    return failures, warnings


def _scan_api():
    if not API_ROOT.exists():
        return [], []
    paths = sorted(API_ROOT.rglob("*.py"))
    return _scan_with(ApiScopeChecker, paths)


def _scan_runtime():
    if not RUNTIME_ROOT.exists():
        return [], []
    failures: list[str] = []
    warnings: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f"{path}:{e.lineno}: SyntaxError: {e.msg}")
            continue
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        docstring_ids = _docstring_node_ids(tree)
        checker = RuntimeScopeChecker(rel, docstring_ids)
        checker.visit(tree)
        failures.extend(checker.failures)
        warnings.extend(checker.warnings)
    return failures, warnings


def _scan_adapters():
    """Returns (failures, warnings). DAG-05 cross-arch imports."""
    if not ADAPTERS_ROOT.exists():
        return [], []
    failures: list[str] = []
    for path in sorted(ADAPTERS_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        # Determine the file's arch.
        # adapters/<arch>.py -> arch = stem
        # adapters/<arch>/<file>.py -> arch = directory name (Pi05 siblings exempt)
        # adapters/__init__.py -> arch = None
        parent = path.parent
        if parent == ADAPTERS_ROOT:
            if path.stem == "__init__":
                file_arch = None
            else:
                file_arch = path.stem
        else:
            # Inside an adapter subpackage like adapters/pi05/.
            file_arch = parent.name
            # Pi05 siblings are allowed to cross-import each other (single arch).
            if file_arch == "pi05" and path.name in PI05_SIBLINGS:
                # File arch stays "pi05" — same-arch imports OK; cross-arch still fail.
                pass
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f"{rel}:{e.lineno}: SyntaxError: {e.msg}")
            continue
        checker = AdapterCrossArchChecker(rel, file_arch)
        checker.visit(tree)
        failures.extend(checker.failures)
    return failures, []


# ============================================================================
# Self-test fixtures (HND-05)
# ============================================================================

# Each fixture: (filename relative to runtime/ or adapters/, source).
# `expect`: 'pass' or 'fail'.
SELF_TEST_FIXTURES = [
    # POSITIVE (must pass full-mode scan).
    ('runtime/hw_attrs.py', '_DECODER_HANDLE = "_rpu_decoder_handle"\n', 'pass'),
    ('runtime/hw_attrs.py', '_VLM_DECODER_HANDLE = "_rpu_vlm_decoder_handle"\n', 'pass'),
    ('runtime/hw_attrs.py', '_ACTION_HANDLE = "_rpu_action_handle"\n', 'pass'),
    ('runtime/hw_attrs.py', '_VISION_HANDLE = "_rpu_vision_handle"\n', 'pass'),
    ('runtime/hw_attrs.py', '_CHUNK_SIZE_ATTR = "_rpu_chunk_size"\n', 'pass'),
    ('runtime/decoder.py', 'def _install_rotary_class_swap(rotary_emb_class):\n    pass\n', 'pass'),
    # Pi05 sibling import (allowed)
    ('adapters/pi05/gemma.py',
     'from rpu_backend.adapters.pi05.adarms import patch_adarms_model_for_rpu_all_layers_once\n',
     'pass'),
    # Shared SigLIP peer imports are allowed only for multimodal policy adapters.
    ('adapters/pi05/__init__.py',
     'from rpu_backend.adapters.siglip import patch_siglip_model_for_rpu_all_layers_once\n',
     'pass'),
    ('adapters/paligemma/__init__.py',
     'from rpu_backend.adapters.siglip import patch_siglip_model_for_rpu_all_layers_once\n',
     'pass'),
    # HALO's ViT IS SigLIP-SO400M-navit — vit_cache reuses the siglip weight converters.
    ('adapters/halo/vit_cache.py',
     'from rpu_backend.adapters.siglip import _convert_siglip_weights_for_rpu\n',
     'pass'),
    # GR00T-N1.7's backbone IS Qwen3-VL-2B — the gr00t policy reuses the qwen3_vl peer.
    ('adapters/gr00t/runtime.py',
     'from rpu_backend.adapters.qwen3_vl import install_qwen3_vl_vision_for_rpu\n',
     'pass'),
    ('adapters/lingbot_vla_v2/runtime.py',
     'from rpu_backend.adapters.qwen3_vl import install_qwen3_vl_vision_for_rpu\n',
     'pass'),
    ('adapters/rhinovla/vision.py',
     'from rpu_backend.adapters.qwen3_vl.vision import _quantize_vision_w8_weight\n',
     'pass'),
    ('adapters/rhinovla/vision.py',
     'from rpu_backend.adapters.qwen3_vl import install_qwen3_vl_vision_for_rpu\n',
     'fail'),
    ('adapters/qwen3_vl/vision.py',
     'from rpu_backend.adapters.rhinovla.vision import patch_qwen3vl_vision_for_rpu\n',
     'fail'),
    # G0.5's VLM and action expert ARE Qwen3.5 components.
    ('adapters/g05/runtime.py',
     'from rpu_backend.adapters.qwen3_5.text import run_qwen3_5_text\n',
     'pass'),
    # InternVLA System-2 shares Wall-OSS's bounded-memory causal-decoder loader.
    ('adapters/internvla_n1/backbone.py',
     'from rpu_backend.adapters.wall_oss.llm import _SafeTensorStore\n',
     'pass'),
    # NavDP is the InternVLA System-1 head and must use the same controlled policy.
    ('adapters/navdp/runtime.py',
     'from rpu_backend.adapters.internvla_n1._policy import require_controlled_evaluation\n',
     'pass'),
    # Qwen3.5-MoE shares the dense Vision tower; Vision verifies MoE owner identity.
    ('adapters/qwen3_5/vision.py',
     'from rpu_backend.adapters.qwen3_5_moe import _close_qwen3_5_moe_model\n',
     'pass'),
    ('adapters/qwen3_5/vision.py',
     'from rpu_backend.adapters.qwen3_5_moe.cores import validate_native_topology\n',
     'pass'),
    ('adapters/qwen3_5_moe/cores.py',
     'from rpu_backend.adapters.qwen3_5.cores import topology_tuple\n',
     'pass'),
    ('adapters/qwen3_5_moe/vision.py',
     'from rpu_backend.adapters.qwen3_5.vision import install_qwen3_5_vision_for_rpu\n',
     'pass'),
    # NEGATIVE (must fail full-mode scan).
    ('runtime/hw_attrs.py', '_OLD_QWEN_HANDLE = "_rpu_qwen3_handle"\n', 'fail'),
    ('runtime/hw_attrs.py', '_OLD_GEMMA_HANDLE = "_rpu_gemma_handle"\n', 'fail'),
    ('runtime/hw_attrs.py', '_OLD_ADARMS_HANDLE = "_rpu_adarms_handle"\n', 'fail'),
    ('runtime/hw_attrs.py', '_OLD_SIGLIP_HANDLE = "_rpu_siglip_handle"\n', 'fail'),
    ('runtime/control.py', 'from rpu_backend.adapters.qwen3 import Qwen3Adapter\n', 'fail'),
    ('runtime/control.py', '_OP = f"qwen3_op"\n', 'fail'),
    ('runtime/control.py', '_OP = f"qwenpi05_op"\n', 'fail'),
    ('runtime/control.py', '_ARCH = "LlamaForCausalLM"\n', 'fail'),
    ('runtime/control.py', '_PI05_MAGIC = 42\n', 'fail'),
    ('adapters/qwen3.py', 'from rpu_backend.adapters.llama import LlamaAdapter\n', 'fail'),
    ('adapters/paligemma/__init__.py',
     'from rpu_backend.adapters.pi05.gemma import patch_gemma_model_for_rpu_all_layers_once\n',
     'fail'),
    # The Qwen3.5 exception is exact to G0.5, not a general peer bypass.
    ('adapters/wall_oss/runtime.py',
     'from rpu_backend.adapters.qwen3_5.text import run_qwen3_5_text\n',
     'fail'),
    # gr00t<->wall_oss sibling coupling stays forbidden: the shared partial-mRoPE
    # helper lives in runtime/, not wall_oss/ (see ALLOWED_SHARED_PEER_IMPORTS note).
    ('adapters/gr00t/runtime.py',
     'from rpu_backend.adapters.wall_oss.rope_partial import build_interleaved_mrope_cos_sin\n',
     'fail'),
    # The sanctioned NavDP -> InternVLA policy edge must not create a cycle.
    ('adapters/internvla_n1/rgbd_rpu.py',
     'from rpu_backend.adapters.navdp.runtime import RPUCache\n',
     'fail'),
    # The NavDP exception is module-exact, not access to the whole peer adapter.
    ('adapters/navdp/runtime.py',
     'from rpu_backend.adapters.internvla_n1.backbone import Qwen25VLBackbone\n',
     'fail'),
    # The Vision bridge does not authorize adjacent dense/MoE modules.
    ('adapters/qwen3_5/vision.py',
     'from rpu_backend.adapters.qwen3_5_moe.text import run_qwen3_5_moe_text\n',
     'fail'),
    ('adapters/qwen3_5_moe/vision.py',
     'from rpu_backend.adapters.qwen3_5.text import run_qwen3_5_text\n',
     'fail'),
    ('adapters/qwen3_5_moe/text.py',
     'from rpu_backend.adapters.qwen3_5.weights import pad_gdn_scalar\n',
     'fail'),
]


def _classify_fixture_path(virtual_path: str):
    """Map a virtual fixture path like 'runtime/hw_attrs.py' to (checker, args)."""
    parts = virtual_path.split('/')
    if parts[0] == 'runtime':
        return ('runtime', pathlib.Path(virtual_path), None)
    if parts[0] == 'adapters':
        # adapters/<arch>.py or adapters/<arch>/<sibling>.py
        if len(parts) == 2:
            arch = pathlib.Path(parts[1]).stem
        else:
            arch = parts[1]
        return ('adapters', pathlib.Path(virtual_path), arch)
    return ('unknown', pathlib.Path(virtual_path), None)


def _run_self_test() -> int:
    """Exercise each fixture; return 0 if all classifications match expectation."""
    failures: list[str] = []
    for virtual_path, source, expect in SELF_TEST_FIXTURES:
        scope, rel, arch = _classify_fixture_path(virtual_path)
        try:
            tree = ast.parse(source, filename=virtual_path)
        except SyntaxError as e:
            failures.append(f"fixture {virtual_path}: SyntaxError {e}")
            continue
        if scope == 'runtime':
            checker = RuntimeScopeChecker(rel, _docstring_node_ids(tree))
        elif scope == 'adapters':
            checker = AdapterCrossArchChecker(rel, arch)
        else:
            failures.append(f"fixture {virtual_path}: unknown scope")
            continue
        checker.visit(tree)
        actual_fails = list(checker.failures)
        if expect == 'pass' and actual_fails:
            failures.append(
                f"FIXTURE [{expect}→ACTUAL FAIL] {virtual_path!r} "
                f"(source={source.strip()!r})\n  failures: {actual_fails}"
            )
        elif expect == 'fail' and not actual_fails:
            failures.append(
                f"FIXTURE [{expect}→ACTUAL PASS] {virtual_path!r} "
                f"(source={source.strip()!r}) — checker did not flag a violation"
            )
    if failures:
        print("SELF-TEST FAILURES:", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1
    print(f"OK: --self-test ({len(SELF_TEST_FIXTURES)} fixtures)")
    return 0


# ============================================================================
# Entry point
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "v5 DAG checker. Default = full mode (api/ + runtime/ + adapters/ scope, "
            "v5-04 expansion of the v5-01a-1 minimal mode). Use --self-test for "
            "fixture-based regression."
        )
    )
    parser.add_argument(
        "--self-test", action="store_true", default=False,
        help="Run inline positive/negative fixtures; assert all classifications match.",
    )
    # v5-04 codex Stage-4 R1 M-02 closure: keep `--minimal-mode` and `--full-mode`
    # flags advertised so locked-decision D-08 ("keep --minimal-mode flag for backward
    # compat") is honored. Both currently route to the same default-mode scan; the
    # SKEL-07 v5-01a-1 minimal-only behavior was superseded by v5-04 expansion (the
    # full-mode scan IS the minimal-mode scan plus runtime/ + adapters/ extensions).
    parser.add_argument(
        "--full-mode", action="store_true", default=False,
        help="Explicit full mode (default). v5-04 superset of v5-01a-1 SKEL-07 minimal mode.",
    )
    parser.add_argument(
        "--minimal-mode", action="store_true", default=False,
        help=(
            "v5-01a-1 SKEL-07 backward-compat alias for the api/-only scan. v5-04 "
            "merged minimal into the default full-mode scan; this flag now runs the "
            "full scan and is preserved per locked decision D-08."
        ),
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_test()

    api_fails, api_warns = _scan_api()
    runtime_fails, runtime_warns = _scan_runtime()
    adapter_fails, adapter_warns = _scan_adapters()

    all_warnings = api_warns + runtime_warns + adapter_warns
    all_failures = api_fails + runtime_fails + adapter_fails

    for w in all_warnings:
        print(f"WARN: {w}", file=sys.stderr)
    if all_failures:
        for f in all_failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print(
        f"OK: api/ + runtime/ + adapters/ DAG checks "
        f"({len(all_warnings)} warnings; full mode v5-04)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
