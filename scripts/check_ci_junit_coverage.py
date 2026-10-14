#!/usr/bin/env python3
"""Check that pytest commands publish their JUnit output, including on failure."""
from __future__ import annotations

import copy
import fnmatch
import pathlib
import re
import shlex
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_ci_needs import _GitLabLoader  # noqa: E402  the one GitLab YAML loader

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CI_FILE = REPO_ROOT / ".gitlab-ci.yml"

_PYTEST = ("-m", "pytest")


class _Reference(tuple):
    """Keep the existing GitLab tag's path until its consumer is known."""


class _CoverageLoader(_GitLabLoader):
    pass


_CoverageLoader.add_constructor(
    "!reference", lambda loader, node: _Reference(loader.construct_sequence(node)))


def _effective_jobs(doc):
    """Resolve GitLab map inheritance and script references in this checker.

    Arrays replace rather than concatenate under extends. A referenced script
    brings commands, never the referenced template's artifacts, into its job.
    """
    resolved = {}

    def merge(base, overlay):
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            result[key] = (merge(result[key], value)
                           if isinstance(result.get(key), dict) and isinstance(value, dict)
                           else copy.deepcopy(value))
        return result

    def job(name, chain=()):
        if name in chain:
            raise ValueError("cyclic extends: " + " -> ".join((*chain, name)))
        if name in resolved:
            return resolved[name]
        spec = doc.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"missing extends/reference job {name}")
        value = {}
        parents = spec.get("extends") or []
        if isinstance(parents, str):
            parents = [parents]
        for parent in parents:
            value = merge(value, job(parent, (*chain, name)))
        value = merge(value, spec)
        resolved[name] = value
        return value

    def reference(value, chain=()):
        if isinstance(value, _Reference):
            path = tuple(value)
            if not path or path in chain:
                raise ValueError(f"missing or cyclic !reference {path}")
            result = job(path[0])
            for key in path[1:]:
                if not isinstance(result, dict) or key not in result:
                    raise ValueError(f"missing !reference {path}")
                result = result[key]
            return reference(result, (*chain, path))
        if isinstance(value, list):
            return [reference(item, chain) for item in value]
        if isinstance(value, dict):
            return {key: reference(item, chain) for key, item in value.items()}
        return value

    for name, spec in doc.items():
        if isinstance(spec, dict) and name not in {"default", "variables", "workflow", "include"}:
            effective = job(name)
            # Only resolve the fields involved in test execution/reporting.
            # Variables and rules can contain unrelated GitLab expressions.
            yield name, {key: reference(effective[key]) for key in
                         ("script", "before_script", "after_script", "artifacts") if key in effective}


def _script_lines(spec) -> list:
    """Flatten referenced/nested script arrays without dropping any command."""
    out = []

    def visit(value):
        if isinstance(value, str):
            out.extend(value.splitlines())
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif value is not None:
            raise ValueError("script/reference must contain strings or script arrays")

    for key in ("before_script", "script", "after_script"):
        visit(spec.get(key))
    return out


def _invocations(lines: list) -> list:
    """The `-m pytest` commands in one job, each as a token list.

    Two shapes have to survive:

      * a YAML `- |` block joins continuation lines with a newline plus a
        trailing backslash, so one invocation spans several source lines;
      * most of the hw invocations are wrapped in `flock ... -c "<command>"`,
        which shlex reduces to a SINGLE token. Missing those is not academic —
        8 of the 11 pytest calls in this file are inside a flock, and a
        first draft of this checker passed a tree where one of them had had its
        --junitxml removed.

    So: join the backslashes, shlex, and recurse into any token that still
    looks like a whole command. Not a bare regex over the raw text — that reads
    `-m pytest` out of comments and out of quoted prose, and this file has both.
    """
    joined, buffer = [], ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        joined.append(buffer + stripped)
        buffer = ""
    if buffer:
        joined.append(buffer)

    found, pending = [], list(joined)
    while pending:
        command = pending.pop()
        try:
            tokens = shlex.split(command, comments=True)
        except ValueError:                       # unbalanced quotes: shell-only
            continue
        for i, token in enumerate(tokens):
            if tuple(tokens[i:i + 2]) == _PYTEST:
                found.append(tokens[i + 2:])
            elif token == "scripts/ci/run_host_tests.sh":
                found.extend(_invocations((REPO_ROOT / token).read_text().splitlines()))
            elif token == "scripts/ci/run_host_tests.py" and i and "python" in tokens[0]:
                found.append(tokens[i + 1:])
            elif i and tokens[i - 1] == "-c":
                # `flock ... -c "<command>"`. Keyed on the -c and nothing else:
                # "any quoted token mentioning pytest" also matches prose like
                # `echo "run -m pytest to reproduce"`.
                pending.append(token)
    return found


def _has_junit_report(spec) -> bool:
    reports = (spec.get("artifacts") or {}).get("reports") or {}
    return bool(reports.get("junit"))


def _uploads_on_failure(spec) -> bool:
    """Does the reporting stanza survive a red job? (assertion 4)"""
    return str((spec.get("artifacts") or {}).get("when") or "on_success") == "always"


def _report_resolver(lines):
    """Resolve literal shell assignments and the existing $1 report wrapper.

    This never executes shell text. Unknown directory expressions fail closed;
    variables within a filename remain symbolic and must fit the upload glob.
    """
    assignments, functions, active = {}, {}, None
    for line in lines:
        match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z_0-9]*)\(\)\s*\{\s*", line)
        if match:
            active = match[1]
        if line.strip() == "}":
            active = None
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        if tokens[:1] in (["local"], ["export"]):
            tokens = tokens[1:]
        if len(tokens) == 1 and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", tokens[0]):
            key, value = tokens[0].split("=", 1)
            assignments.setdefault(key, set()).add(value)
            if active and re.fullmatch(r"\$[1-9]", value):
                functions[key] = (active, int(value[1:]))

    def expand(value, seen=()):
        match = re.match(r"^\$(?:\{([A-Za-z_][A-Za-z_0-9]*)\}|([A-Za-z_][A-Za-z_0-9]*))", value)
        if not match:
            return [value.removeprefix("./")]
        key = match[1] or match[2]
        suffix = value[match.end():]
        if key == "CI_PROJECT_DIR":
            return [suffix.lstrip("/")]
        if key in seen:
            raise ValueError(f"cyclic JUnit output variable {key}")
        values = assignments.get(key, set())
        if key in functions:
            function, index = functions[key]
            values = set()
            for line in lines:
                try:
                    args = shlex.split(line.rstrip().removesuffix("\\"), comments=True)
                except ValueError:
                    continue
                if args and args[0] == function and len(args) > index:
                    values.add(args[index])
        if not values:
            raise ValueError(f"unresolved JUnit output directory {value}")
        return [path for item in sorted(values) for path in expand(item + suffix, (*seen, key))]

    return expand


def _report_matches(path, pattern):
    # GitLab artifact globs do not let a single * cross a directory boundary.
    # Preserve ** for recursive upload patterns, including zero directories.
    def match(parts, globs):
        if not globs:
            return not parts
        if globs[0] == "**":
            return match(parts, globs[1:]) or bool(parts and match(parts[1:], globs))
        return bool(parts and fnmatch.fnmatchcase(parts[0], globs[0])
                    and match(parts[1:], globs[1:]))
    return match(path.removeprefix("./").split("/"), pattern.removeprefix("./").split("/"))


def check(doc: dict) -> list:
    problems = []
    try:
        jobs = list(_effective_jobs(doc))
    except (ValueError, TypeError) as error:
        return [str(error)]
    for name, spec in jobs:
        try:
            lines = _script_lines(spec)
        except ValueError as error:
            problems.append(f"{name}: {error}")
            continue
        invocations = [args for args in _invocations(lines) if "--collect-only" not in args]
        if not invocations:
            continue
        outputs = []
        for args in invocations:
            flags = [a for a in args if a == "--junitxml" or a.startswith("--junitxml=")]
            if not flags:
                problems.append(f"{name}: a `-m pytest` invocation has no "
                                f"--junitxml (args: {' '.join(args[:4])} ...)")
                continue
            if len(flags) != 1:
                problems.append(f"{name}: pytest requires one unambiguous --junitxml output")
                continue
            index = args.index(flags[0])
            output = (flags[0].split("=", 1)[1] if "=" in flags[0] else
                      args[index + 1] if index + 1 < len(args) else "")
            if not output:
                problems.append(f"{name}: --junitxml requires an output path")
            else:
                outputs.append(output)
        # A hidden snippet is still checked for malformed pytest commands.
        # Its reports belong to each real consumer, including inherited jobs.
        if name.startswith("."):
            continue
        if not _has_junit_report(spec):
            problems.append(f"{name}: runs pytest but declares no "
                            f"`artifacts: reports: junit:`")
            continue
        if not _uploads_on_failure(spec):
            problems.append(
                f"{name}: declares `artifacts: reports: junit:` but not "
                f"`artifacts: when: always` (GitLab defaults to on_success, so "
                f"a FAILED run would upload no report at all)")
        reports = spec["artifacts"]["reports"]["junit"]
        if isinstance(reports, str):
            reports = [reports]
        if not isinstance(reports, list) or any(not isinstance(item, str) for item in reports):
            problems.append(f"{name}: JUnit reports must be paths or globs")
            continue
        resolve_report = _report_resolver(lines)
        for output in outputs:
            try:
                for path in resolve_report(output):
                    if not any(_report_matches(path, pattern) for pattern in reports):
                        problems.append(f"{name}: JUnit output {path} is not covered by reports:junit")
            except ValueError as error:
                problems.append(f"{name}: {error}")
    return problems


def _self_test() -> int:
    REPORT = {"reports": {"junit": "junit/*.xml"}, "when": "always"}

    ok = {"j": {"script": ["python -m pytest -q --junitxml=junit/j.xml tests/a.py"],
                "artifacts": dict(REPORT)}}
    assert not check(ok), check(ok)

    launcher = {"j": {"script": ["python -B scripts/ci/run_host_tests.py --core --junitxml=junit/core.xml"],
                        "artifacts": dict(REPORT)}}
    assert not check(launcher), check(launcher)
    launcher["j"]["script"] = ["python -B scripts/ci/run_host_tests.py --core"]
    assert check(launcher), "the host launcher must publish its actual report"

    missing_flag = {"j": {"script": ["python -m pytest -q tests/a.py"],
                          "artifacts": dict(REPORT)}}
    assert check(missing_flag), "a pytest call with no --junitxml must fail"

    collect_only = {
        "j": {"script": ["python -m pytest --collect-only -q tests/a.py"]}
    }
    assert not check(collect_only), "collection executes no tests to report"

    missing_report = {"j": {"script": ["python -m pytest --junitxml=j.xml t.py"]}}
    assert check(missing_report), "a pytest job with no reports:junit must fail"

    after_positional = {
        "j": {"script": ["python -m pytest -q tests/a.py --junitxml=junit/j.xml"],
              "artifacts": dict(REPORT)}}
    assert not check(after_positional), "pytest accepts report flags after test paths"

    # An option VALUE is not a test path. `-p no:cacheprovider` used to be read
    # as the first positional and failed a correct command (.hw_correct_base).
    value_opt = {
        "j": {"script": ["python -m pytest -q --no-header -p no:cacheprovider "
                         "--junitxml=junit/j.xml tests/a.py"],
              "artifacts": dict(REPORT)}}
    assert not check(value_opt), check(value_opt)
    # Options after test paths are also valid pytest syntax.
    value_opt_bad = {
        "j": {"script": ["python -m pytest -q -p no:cacheprovider tests/a.py "
                         "--junitxml=junit/j.xml"],
              "artifacts": dict(REPORT)}}
    assert not check(value_opt_bad), check(value_opt_bad)

    # Assertion 4. GitLab's default is on_success, so BOTH an absent `when:` and
    # an explicit non-always value lose the report on exactly the run that
    # needed it.
    no_when = {"j": {"script": ["python -m pytest -q --junitxml=junit/j.xml t.py"],
                     "artifacts": {"reports": {"junit": "junit/*.xml"}}}}
    assert check(no_when), "a reporting job without `when: always` must fail"
    on_failure = {"j": {"script": ["python -m pytest -q --junitxml=junit/j.xml t.py"],
                        "artifacts": {"reports": {"junit": "junit/*.xml"},
                                      "when": "on_success"}}}
    assert check(on_failure), "`when: on_success` on a reporting job must fail"

    # A `- |` block whose invocation spans continuation lines is ONE call.
    block = {"j": {"script": ["timeout 900 sudo -n env X=1 \\\n"
                              "  python -m pytest -q --junitxml=junit/j.xml \\\n"
                              "  tests/a.py \\\n  tests/b.py"],
                   "artifacts": dict(REPORT)}}
    assert not check(block), check(block)

    # `flock ... -c "<command>"` is ONE shlex token; recurse or miss 8 of 11.
    wrapped = {"j": {"script": ['flock /var/lock/test.device -c "timeout 600 '
                                'sudo -n env X=1 python -m pytest -q t.py"'],
                     "artifacts": dict(REPORT)}}
    assert check(wrapped), "a flock-wrapped pytest with no --junitxml must fail"
    wrapped_ok = {"j": {"script": ['flock /var/lock/test.device -c "timeout 600 '
                                   'python -m pytest -q --junitxml=junit/j.xml t.py"'],
                        "artifacts": dict(REPORT)}}
    assert not check(wrapped_ok), check(wrapped_ok)

    # The two false positives a bare regex produces on this very file.
    quoted = {"j": {"script": ['echo "run -m pytest by hand to reproduce"']}}
    assert not check(quoted), f"quoted prose is not an invocation: {check(quoted)}"
    commented = {"j": {"script": ["# `python -m pytest`, never script invocation.",
                                  "python scripts/check_no_dup.py"]}}
    assert not check(commented), f"a comment is not an invocation: {check(commented)}"

    # Hidden snippets still have their pytest commands checked even unused.
    template = {".base": {"script": ["python -m pytest -q t.py"]}}
    assert check(template), "hidden pytest commands must retain their JUnit flag"

    def pipeline(text):
        return yaml.load(text, Loader=_CoverageLoader)

    # !reference copies only script; two independent consumers must upload it.
    referenced = pipeline("""
.snippet:
  script:
    - python -m pytest --junitxml=junit/nested.xml tests/a.py
.chain:
  script:
    - !reference [.snippet, script]
.first:
  script:
    - !reference [.chain, script]
  artifacts:
    reports: {junit: 'junit/*.xml'}
    when: always
job:
  extends: .first
other:
  script:
    - !reference [.chain, script]
  artifacts:
    reports: {junit: 'junit/*.xml'}
    when: always
""")
    assert not check(referenced), check(referenced)
    for change in ("report", "when", "destination", "flag", "reference", "cycle"):
        broken = copy.deepcopy(referenced)
        if change == "report":
            broken["other"]["artifacts"]["reports"] = None
        elif change == "when":
            del broken["other"]["artifacts"]["when"]
        elif change == "destination":
            broken["other"]["artifacts"]["reports"]["junit"] = "elsewhere/*.xml"
        elif change == "flag":
            broken[".snippet"]["script"] = ["python -m pytest tests/a.py"]
        elif change == "reference":
            broken[".chain"]["script"] = [_Reference((".absent", "script"))]
        else:
            broken[".chain"]["script"] = [_Reference((".first", "script"))]
        assert check(broken), f"reference consumer {change} must fail"

    # A child replaces the inherited script array, while nested artifact maps
    # merge. Dropping a sibling consumer's report cannot borrow the parent's.
    replaced = copy.deepcopy(referenced)
    replaced["job"].update(script=["echo no-tests"], artifacts=None)
    assert not check(replaced), check(replaced)
    overridden = copy.deepcopy(referenced)
    overridden["job"]["artifacts"] = {"reports": {"junit": "wrong.xml"}}
    assert any("job: JUnit output" in error for error in check(overridden))
    inherited = {".base": {"script": ["python -m pytest --junitxml=junit/a.xml a.py"]},
                 "job": {"extends": ".base"}}
    assert any("job: runs pytest" in error for error in check(inherited))
    dangling = {"job": {"extends": ".missing"}}
    assert check(dangling), "missing inheritance must not erase a test producer"
    cyclic = {".a": {"extends": ".b"}, ".b": {"extends": ".a"}}
    assert check(cyclic), "cyclic inheritance must fail"

    # Existing shell wrapper: validate every concrete output, including a
    # report directory mismatch on only the second call.
    wrapper = {"job": {"script": ["""run_suite() {
  local report=$1
  python -m pytest --junitxml="$report" tests/a.py
}
run_suite junit/a.xml
run_suite junit/b.xml
"""], "artifacts": dict(REPORT)}}
    assert not check(wrapper), check(wrapper)
    wrapper["job"]["script"][0] = wrapper["job"]["script"][0].replace("junit/b.xml", "lost/b.xml")
    assert any("lost/b.xml" in error for error in check(wrapper))
    variable = {"job": {"script": ['REPORT="junit/run-$CASE.xml"',
                        'python -m pytest --junitxml "$REPORT" tests/a.py'],
                        "artifacts": dict(REPORT)}}
    assert not check(variable), check(variable)
    variable["job"]["script"][0] = 'REPORT="lost/run-$CASE.xml"'
    assert check(variable), "literal variable expansion must not lose output coverage"

    assert not _report_matches("junit/deep/lost.xml", "junit/*.xml")
    assert not _report_matches("elsewhere/junit/a.xml", "junit/*.xml")
    assert _report_matches("junit/deep/a.xml", "junit/**/*.xml")
    assert _report_matches("junit/a.xml", "junit/**/*.xml")
    duplicate = copy.deepcopy(ok)
    duplicate["j"]["script"][0] = "python -m pytest --junitxml=junit/a.xml --junitxml=lost.xml a.py"
    assert check(duplicate), "pytest's last output flag must not evade report coverage"

    print("check_ci_junit_coverage self-test: OK")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()
    doc = yaml.load(CI_FILE.read_text(encoding="utf-8"), Loader=_CoverageLoader)
    problems = check(doc)
    if problems:
        print("FAIL: pytest invocations not wired to a JUnit report")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASS: every test-executing `-m pytest` invocation writes JUnit, "
          "every executing consumer publishes the matching output, and every such stanza is "
          "`when: always`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
