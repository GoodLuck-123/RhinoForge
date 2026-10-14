#!/usr/bin/env python3
"""scripts/check_component_spm_contract.py — v5-03 D-06 SPM-reset enforcer."""
from __future__ import annotations
import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Files governed by this verifier. Strict repo-relative allow-list — adding
# new component adapters MUST come with an explicit entry here.
# Each tuple: (patch fn name, fused-op suffix). The fused-op suffix MUST
# match the literal `torch.ops.rpu.<suffix>(...)` Attribute chain emitted
# by the lifted forward body (R1 MEDIUM-2).
TARGET_FILES = {
    "python/rpu_backend/adapters/pi05/gemma.py": (
        "patch_gemma_model_for_rpu_all_layers_once",
        "gemma_forward",
    ),
    "python/rpu_backend/adapters/pi05/adarms.py": (
        "patch_adarms_model_for_rpu_all_layers_once",
        "adarms_forward",
    ),
    "python/rpu_backend/adapters/siglip.py": (
        "patch_siglip_model_for_rpu_all_layers_once",
        "siglip_forward",
    ),
    "python/rpu_backend/adapters/paligemma/gemma2.py": (
        "patch_gemma2_model_for_rpu_all_layers_once",
        "gemma2_forward",
    ),
}


def _is_torch_ops_rpu_call(node: ast.AST, expected_attr: str) -> bool:
    """True iff `node` is a Call whose `.func` is the Attribute chain
    `torch.ops.rpu.<expected_attr>`. Guards against false positives where
    another object happens to expose an attribute of the same name."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    a3 = node.func                              # torch.ops.rpu.<attr>
    if a3.attr != expected_attr:
        return False
    a2 = a3.value                                # torch.ops.rpu
    if not isinstance(a2, ast.Attribute) or a2.attr != "rpu":
        return False
    a1 = a2.value                                # torch.ops
    if not isinstance(a1, ast.Attribute) or a1.attr != "ops":
        return False
    a0 = a1.value                                # torch
    return isinstance(a0, ast.Name) and a0.id == "torch"


def _find_first_call_lineno(fn: ast.FunctionDef, expected_attr: str) -> int | None:
    """Walk fn body; return the lineno of the first Call matching
    `torch.ops.rpu.<expected_attr>(...)` or None if absent."""
    return min((node.lineno for node in _scope_nodes(fn)
                if _is_torch_ops_rpu_call(node, expected_attr)), default=None)


def _scope_nodes(fn: ast.FunctionDef):
    """Inspect this executed body, excluding uncalled nested definitions."""
    pending = list(fn.body)
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))


def _scope_bindings(fn: ast.FunctionDef, name: str) -> list[ast.AST]:
    bindings = []
    for node in _scope_nodes(fn):
        if ((isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
             and node.id == name)
                or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == name)
                or (isinstance(node, ast.alias)
                    and (node.asname or node.name.split('.')[0]) == name)
                or (isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names)
                or (isinstance(node, ast.ExceptHandler) and node.name == name)):
            bindings.append(node)
    bindings.extend(arg for arg in (
        *fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs,
        *([fn.args.vararg] if fn.args.vararg else []),
        *([fn.args.kwarg] if fn.args.kwarg else [])) if arg.arg == name)
    return bindings


def _check_pi05_prefill_helper(patch_fn: ast.FunctionDef) -> str | None:
    """Follow only the two reviewed Pi0.5 entries and their finite op alias.

    This is not general interprocedural inference: a renamed/rebound helper,
    escaped alias or different op requires review instead of earning a reset
    credit merely because the expected op appears somewhere in the source.
    """
    helper_name = "_run_gemma_prefill"
    helpers = [node for node in patch_fn.body
               if isinstance(node, ast.FunctionDef) and node.name == helper_name]
    if len(helpers) != 1 or _scope_bindings(patch_fn, helper_name) != helpers:
        return "shared prefill helper is missing or rebound"
    helper = helpers[0]
    mode_bindings = _scope_bindings(helper, "_kv_only")
    if (len(mode_bindings) != 1 or not isinstance(mode_bindings[0], ast.arg)
            or _scope_bindings(helper, "torch") or _scope_bindings(patch_fn, "torch")):
        return "shared prefill mode or torch receiver is rebound"
    for entry_name, kv_only in (("rpu_gemma_model_forward", False),
                                ("rpu_gemma_prefill_kv_only", True)):
        entries = [node for node in patch_fn.body
                   if isinstance(node, ast.FunctionDef) and node.name == entry_name]
        if len(entries) != 1 or _scope_bindings(patch_fn, entry_name) != entries:
            return f"shared prefill entry {entry_name} is missing or rebound"
        entry = entries[0]
        body = list(entry.body)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            return f"{entry_name} must directly return the shared prefill call"
        call = body[0].value
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id != helper_name or _scope_bindings(entry, helper_name)):
            return f"{entry_name} has a broken shared prefill callee chain"
        modes = [kw.value for kw in call.keywords if kw.arg == "_kv_only"]
        if (len(modes) != 1 or not isinstance(modes[0], ast.Constant)
                or modes[0].value is not kv_only):
            return f"{entry_name} selects the wrong shared prefill mode"

    alias = "native_forward"
    bindings = [node for node in helper.body if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == alias]
    if len(bindings) != 1 or _scope_bindings(helper, alias) != bindings[0].targets:
        return "shared prefill native alias is missing or rebound"
    binding = bindings[0]
    expected = ast.parse(
        "torch.ops.rpu.gemma_prefill_kv_only if _kv_only else torch.ops.rpu.gemma_forward",
        mode="eval").body
    if ast.dump(binding.value) != ast.dump(expected):
        return "shared prefill native alias must select the exact two Gemma ops"
    calls = [node for node in _scope_nodes(helper) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == alias]
    loads = [node for node in _scope_nodes(helper) if isinstance(node, ast.Name)
             and isinstance(node.ctx, ast.Load) and node.id == alias]
    if len(calls) != 2 or {id(node) for node in loads} != {id(call.func) for call in calls}:
        return "shared prefill native alias is not called or escapes its body"
    if min(call.lineno for call in calls) <= binding.lineno:
        return "shared prefill native alias is called before its binding"
    reset_lineno = _find_first_call_lineno(helper, "spm_alloc_reset_temporary")
    if reset_lineno is None or reset_lineno <= max(call.lineno for call in calls):
        return "shared prefill reset must follow every native prefill call"
    # The reset belongs to the helper body, outside GraphCache.capture().
    if not any(isinstance(node, ast.Expr)
               and _is_torch_ops_rpu_call(node.value, "spm_alloc_reset_temporary")
               and node.lineno == reset_lineno for node in helper.body):
        return "shared prefill reset must stay outside the capture scope"
    return None


def _find_top_level_fn(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _check_file(
    rel_filename: str,
    patch_fn_name: str,
    fused_op_attr: str,
    failures: list[str],
) -> None:
    path = REPO_ROOT / rel_filename
    relpath = path.relative_to(REPO_ROOT)
    if not path.exists():
        failures.append(f"{relpath}: file not found")
        return
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        failures.append(f"{relpath}:{exc.lineno}: SyntaxError: {exc.msg}")
        return

    patch_fn = _find_top_level_fn(tree, patch_fn_name)
    if patch_fn is None:
        failures.append(
            f"{relpath}: top-level function `{patch_fn_name}` not found"
        )
        return

    # Find every nested FunctionDef whose name starts with `rpu_` and ends in
    # `forward`. Each MUST: (a) call torch.ops.rpu.<fused_op_attr>(...) AT
    # LEAST ONCE, AND (b) call torch.ops.rpu.spm_alloc_reset_temporary() at a
    # lineno strictly greater than the first fused-op call (R1 MEDIUM-2 — the
    # subsystem-boundary reset only protects callers when it fires AFTER the
    # forward op; reset-before-forward would silently regress and bypass this
    # check without the ordering constraint).
    nested_forwards = [
        n for n in ast.walk(patch_fn)
        if isinstance(n, ast.FunctionDef)
        and n is not patch_fn
        and n.name.startswith("rpu_")
        and n.name.endswith("forward")
    ]
    if not nested_forwards:
        failures.append(
            f"{relpath}:{patch_fn.lineno}: `{patch_fn_name}` defines no "
            f"`rpu_*_forward` closure (expected at least one)"
        )
        return

    for inner in nested_forwards:
        fused_lineno = _find_first_call_lineno(inner, fused_op_attr)
        reset_lineno = _find_first_call_lineno(inner, "spm_alloc_reset_temporary")
        if (fused_lineno is None and fused_op_attr == "gemma_forward"
                and patch_fn_name == "patch_gemma_model_for_rpu_all_layers_once"
                and inner.name == "rpu_gemma_model_forward"):
            problem = _check_pi05_prefill_helper(patch_fn)
            if problem is not None:
                failures.append(f"{relpath}:{inner.lineno}: {problem}")
            continue
        if fused_lineno is None:
            failures.append(
                f"{relpath}:{inner.lineno}: nested `{inner.name}` body does "
                f"not call `torch.ops.rpu.{fused_op_attr}(...)` — fused-op "
                f"emission missing from lifted closure"
            )
            continue
        if reset_lineno is None:
            failures.append(
                f"{relpath}:{inner.lineno}: nested `{inner.name}` body does "
                f"not call `torch.ops.rpu.spm_alloc_reset_temporary()` — v3 "
                f"SPM-reset postcondition violated (D-06)"
            )
            continue
        if reset_lineno <= fused_lineno:
            failures.append(
                f"{relpath}:{inner.lineno}: nested `{inner.name}` calls "
                f"spm_alloc_reset_temporary() at line {reset_lineno} which is "
                f"NOT after the fused `torch.ops.rpu.{fused_op_attr}` call at "
                f"line {fused_lineno} — v3 subsystem-boundary postcondition "
                f"violated (R1 MEDIUM-2: reset must fire AFTER the forward op)"
            )


def main() -> int:
    failures: list[str] = []
    for rel_filename, (fn_name, fused_op_attr) in TARGET_FILES.items():
        _check_file(rel_filename, fn_name, fused_op_attr, failures)
    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1
    checked = ", ".join(TARGET_FILES)
    print(
        f"OK: checked {len(TARGET_FILES)} adapter files "
        f"({checked}) (D-06 v3 SPM-reset; R1 MEDIUM-2 ordered)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
