#!/usr/bin/env python3
"""Assert every `needs:` edge in .gitlab-ci.yml is satisfiable on every trigger."""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CI_FILE = REPO_ROOT / ".gitlab-ci.yml"

# Representative environments for the trigger axes this checkout supports.
# Full validation rejects CI_PLANNER_PHASE=native at its entry point, so that
# invalid combination cannot stand in for either supported web flow.
TRIGGERS = ("merge_request_event", "default_branch", "web", "web_native", "web_full", "web_large", "tag", "tag_rc")


def _trigger_env(trigger: str) -> dict[str, str]:
    env = dict(CI_PIPELINE_SOURCE="push", CI_DEFAULT_BRANCH="main", CI_COMMIT_BRANCH="",
               CI_MERGE_REQUEST_TARGET_BRANCH_NAME="", CI_COMMIT_TAG="",
               CI_COMMIT_REF_PROTECTED="false", CI_PLANNER_PHASE="", CI_VALIDATION="",
               LARGE_FORCE="0")
    if trigger == "merge_request_event":
        env.update(CI_PIPELINE_SOURCE=trigger, CI_MERGE_REQUEST_TARGET_BRANCH_NAME="main")
    elif trigger == "default_branch":
        env.update(CI_COMMIT_BRANCH="main", CI_COMMIT_REF_PROTECTED="true")
    elif trigger.startswith("web"):
        env["CI_PIPELINE_SOURCE"] = "web"
        env["CI_PLANNER_PHASE"] = "native" if trigger == "web_native" else ""
        env["CI_VALIDATION"] = "full" if trigger == "web_full" else ""
        env["LARGE_FORCE"] = "1" if trigger == "web_large" else "0"
    else:
        env["CI_COMMIT_TAG"] = "v1.2.3-rc.1" if trigger == "tag_rc" else "v1.2.3"
        env["CI_COMMIT_REF_PROTECTED"] = "true"
    return env


def _matches(condition: str, env: dict[str, str]) -> bool:
    """Evaluate the small equality/regex rule vocabulary used in this file.

    Unknown syntax/variables fail the check instead of silently omitting jobs.
    This is deliberately not a general GitLab expression interpreter.
    """
    if not condition:
        return True

    def value(token: str) -> str:
        token = token.strip()
        if token.startswith("$"):
            key = token[1:]
            if key not in env:
                raise ValueError(f"unmodelled CI rule variable: {key}")
            return env[key]
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            return token[1:-1]
        raise ValueError(f"unsupported CI rule operand: {token}")

    def atom(expression: str) -> bool:
        match = re.fullmatch(r"\s*(\$\w+)\s*(==|!=|=~|!~)\s*(.+?)\s*", expression)
        if not match:
            return bool(value(expression))
        left, operation, right = match.groups()
        actual = value(left)
        if operation in ("=~", "!~"):
            if not (right.startswith("/") and right.endswith("/")):
                raise ValueError(f"unsupported CI rule regex: {right}")
            matched = bool(re.search(right[1:-1], actual))
            return matched if operation == "=~" else not matched
        equal = actual == value(right)
        return equal if operation == "==" else not equal

    # Evaluate all atoms even after a false result so unsupported syntax never
    # hides behind an earlier condition that happens to be false in this env.
    return any([all([atom(part) for part in arm.split("&&")])
                for arm in condition.split("||")])


def _effective_property(name: str, jobs: dict, key: str, stack=()):
    if name in stack:
        raise ValueError(f"cyclic extends: {stack + (name,)}")
    spec = jobs[name]
    if key in spec:
        return spec[key]
    parents = spec.get("extends") or []
    if isinstance(parents, str):
        parents = [parents]
    for parent in reversed(parents):
        if parent not in jobs:
            raise ValueError(f"{name}: unknown extends parent {parent}")
        inherited = _effective_property(parent, jobs, key, stack + (name,))
        if inherited is not None:
            return inherited
    return None


class _GitLabLoader(yaml.SafeLoader):
    """SafeLoader that tolerates GitLab's `!reference` tag."""


_GitLabLoader.add_multi_constructor("!", lambda loader, suffix, node: None)


def _job_needs(spec: dict) -> list[tuple[str, bool]]:
    """Normalise `needs:` to ``(job, optional)`` pairs."""
    needs = spec.get("needs")
    if not needs:
        return []
    out: list[tuple[str, bool]] = []
    for item in needs:
        if isinstance(item, dict):
            out.append((item["job"], bool(item.get("optional", False))))
        else:
            out.append((item, False))
    return out


def _changes_paths(rule: dict):
    """The rule's `changes:` list, or None when it does not restrict paths."""
    changes = rule.get("changes")
    if changes is None:
        return None
    if isinstance(changes, dict):        # `changes: {paths: [...], compare_to: ...}`
        changes = changes.get("paths") or []
    return frozenset(str(path) for path in changes)


def conditions_for(name: str, jobs: dict) -> dict:
    """Triggers -> path restrictions, respecting the first matching rule.

    Path-filtered allow rules can match different diffs, so union their paths
    until an unconditional matching arm terminates the decision. A later never
    arm cannot retract a prior path match. Path-filtered exclusions require glob
    subtraction and are rejected explicitly rather than guessed.
    """
    rules = _effective_property(name, jobs, "rules") or []
    allowed = {}
    for trigger in TRIGGERS:
        paths_seen = frozenset()
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if not _matches(str(rule.get("if", "")), _trigger_env(trigger)):
                continue
            paths = _changes_paths(rule)
            if rule.get("when") == "never":
                if paths is not None:
                    raise ValueError(f"{name}: path-filtered when:never is not modelled")
                break
            if paths is None:
                allowed[trigger] = None
                break
            paths_seen |= paths
        if trigger not in allowed and paths_seen:
            allowed[trigger] = paths_seen
    return allowed


def triggers_for(name: str, jobs: dict) -> set[str]:
    """Triggers a job runs on (path restrictions dropped)."""
    return set(conditions_for(name, jobs))


def check(doc: dict) -> list[str]:
    jobs = {
        name: spec
        for name, spec in doc.items()
        if isinstance(spec, dict) and name not in {"variables", "default", "stages", "cache"}
    }
    problems = []
    try:
        conditions = {name: conditions_for(name, jobs) for name in jobs}
        needs = {name: _job_needs({"needs": _effective_property(name, jobs, "needs")})
                 for name in jobs}
    except ValueError as error:
        return [str(error)]
    for name, spec in jobs.items():
        if name.startswith("."):
            continue
        mine = conditions[name]
        for dependency, optional in needs[name]:
            if dependency not in jobs:
                problems.append(f"{name}: needs '{dependency}', which is not a job in this file")
                continue
            theirs = conditions[dependency]
            missing = set() if optional else set(mine) - set(theirs)
            if missing:
                problems.append(
                    f"{name}: needs '{dependency}', but on trigger(s) "
                    f"{sorted(missing)} '{dependency}' is not in the pipeline "
                    f"— pipeline creation would fail for ALL jobs"
                )
            if optional:
                # GitLab permits an optional need to be absent for either a
                # whole trigger or this trigger's unmatched `changes:` arm.
                # When present it still orders the jobs; no path containment
                # is required for pipeline creation.
                continue
            for trigger, my_paths in mine.items():
                if trigger in missing or theirs.get(trigger) is None:
                    continue
                extra = {"(any path)"} if my_paths is None else my_paths - theirs[trigger]
                if extra:
                    problems.append(
                        f"{name}: needs '{dependency}', but on trigger "
                        f"'{trigger}' it runs for changes to {sorted(extra)} "
                        f"that '{dependency}' does not — that diff would put "
                        f"'{name}' in the pipeline without '{dependency}' and "
                        f"pipeline creation would fail for ALL jobs"
                    )
    return problems


def _self_test() -> int:
    mr = {"if": '$CI_PIPELINE_SOURCE == "merge_request_event"'}
    web = {"if": '$CI_PIPELINE_SOURCE == "web"'}
    tag = {"if": "$CI_COMMIT_TAG =~ /^v1/"}

    bad = {"a": {"rules": [mr]}, "b": {"rules": [mr, tag], "needs": ["a"]}}
    assert check(bad), "should reject: b runs on tag, a does not"

    good = {"a": {"rules": [mr, tag]}, "b": {"rules": [mr], "needs": ["a"]}}
    assert not check(good), f"should accept, got {check(good)}"

    # `extends` inheritance: c has no rules of its own.
    inherit = {".base": {"rules": [mr, web]}, "a": {"rules": [mr, web]},
               "c": {"extends": ".base", "needs": ["a"]}}
    assert not check(inherit), f"extends should inherit triggers, got {check(inherit)}"

    # `when: never` subtracts, so d does NOT run on web and needing a web-less
    # job is fine.
    never = {"a": {"rules": [mr]},
             "d": {"rules": [{"if": '$CI_PIPELINE_SOURCE == "web"', "when": "never"}, mr],
                   "needs": ["a"]}}
    assert not check(never), f"when:never should subtract, got {check(never)}"

    native = {"if": '$CI_PIPELINE_SOURCE == "web" && $CI_PLANNER_PHASE == "native"'}
    phase = {"regular": {"rules": [{**native, "when": "never"}, web]},
             "sample": {"rules": [native], "needs": ["regular"]}}
    assert triggers_for("regular", phase) == {"web", "web_full", "web_large"}
    assert triggers_for("sample", phase) == {"web_native"}
    assert check(phase), "native web cannot need an excluded regular hardware job"
    phase["regular"]["rules"] = [web]
    assert not check(phase), "a common web prerequisite exists in both phases"

    missing = {"b": {"rules": [mr], "needs": ["nope"]}}
    assert check(missing), "should reject a needs on an unknown job"

    optional_missing_trigger = {
        "a": {"rules": [mr]},
        "b": {"rules": [mr, tag], "needs": [{"job": "a", "optional": True}]},
    }
    assert not check(optional_missing_trigger), (
        "optional need may be absent on one trigger"
    )

    # --- changes: the same trap one level down -----------------------------
    src = dict(mr, changes=["src/**/*"])
    src_and_docs = dict(mr, changes=["src/**/*", "docs/**/*"])

    same = {"a": {"rules": [src]}, "b": {"rules": [src], "needs": ["a"]}}
    assert not check(same), f"identical changes lists are fine, got {check(same)}"

    wider_dependency = {"a": {"rules": [src_and_docs]},
                        "b": {"rules": [src], "needs": ["a"]}}
    assert not check(wider_dependency), (
        f"dependency may be wider than its dependent, got {check(wider_dependency)}")

    wider_dependent = {"a": {"rules": [src]},
                       "b": {"rules": [src_and_docs], "needs": ["a"]}}
    assert check(wider_dependent), "should reject: b runs on docs/ changes, a does not"

    unfiltered_dependency = {"a": {"rules": [mr]},
                             "b": {"rules": [src], "needs": ["a"]}}
    assert not check(unfiltered_dependency), (
        f"an unrestricted dependency is always present, got {check(unfiltered_dependency)}")

    unfiltered_dependent = {"a": {"rules": [src]},
                            "b": {"rules": [mr], "needs": ["a"]}}
    assert check(unfiltered_dependent), "should reject: b runs on any diff, a only on src/"

    # Two arms on the same trigger union their paths.
    two_arms = {"a": {"rules": [src, dict(mr, changes=["docs/**/*"])]},
                "b": {"rules": [src_and_docs], "needs": ["a"]}}
    assert not check(two_arms), f"arms should union, got {check(two_arms)}"

    full = {"if": '$CI_PIPELINE_SOURCE == "web" && $CI_VALIDATION == "full"'}
    split = {"full": {"rules": [full]}, "ordinary": {"rules": [web], "needs": ["full"]}}
    assert check(split), "ordinary web cannot require a full-only prerequisite"
    ordered = {"full": {"rules": [full, dict(web, when="never")]},
               "consumer": {"rules": [full], "needs": ["full"]}}
    assert triggers_for("full", ordered) == {"web_full"}
    assert not check(ordered), "a later exclusion cannot undo first full match"
    ordered["full"]["rules"].reverse()
    assert check(ordered), "an earlier exclusion must win"
    partial = {"a": {"rules": [src, dict(mr, when="never")]},
               "b": {"rules": [src], "needs": ["a"]}}
    assert not check(partial), "later never preserves already admitted paths"
    inherited_need = {".base": {"needs": ["nope"]},
                      "a": {"extends": ".base", "rules": [full]}}
    assert check(inherited_need), "inherited needs cannot bypass DAG validation"
    release = {"final": {"rules": [{"if": '$CI_COMMIT_TAG =~ /^v[0-9]+\\.[0-9]+\\.[0-9]+$/'}]},
               "publish": {"rules": [tag], "needs": ["final"]}}
    assert check(release), "RC must not borrow a final-only prerequisite"
    assert check({"a": {"rules": [{"if": '$UNKNOWN == "full"'}]}})

    print("check_ci_needs self-test: OK")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()
    doc = yaml.load(CI_FILE.read_text(), Loader=_GitLabLoader)
    problems = check(doc)
    if problems:
        print("FAIL: unsatisfiable `needs:` edges in .gitlab-ci.yml")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASS: every `needs:` edge is satisfiable on every trigger its job runs on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
