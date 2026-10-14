# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities through
[GitHub private vulnerability reporting](https://github.com/HUIXI-AI/RhinoForge/security/advisories/new).
Include the affected RhinoForge version, a minimal reproduction, impact, and
any relevant Rhino Launch and operator-asset versions or hashes.

Do not report vulnerabilities in a public issue, discussion, or pull request.
Do not attach restricted binaries, operator assets, model checkpoints, secrets,
personal data, or private infrastructure details. Describe how an authorized
maintainer can reproduce the issue instead.

Maintainers will acknowledge the private report, investigate the supported
release scope, coordinate remediation and disclosure when appropriate, and
publish an advisory for confirmed issues that affect a public release.

## Scope

This policy covers RhinoForge source and release artifacts published by the
HUIXI-AI organization. Separately distributed Rhino Launch packages, operator
assets, model checkpoints, third-party dependencies, and components of the
preconfigured board environment remain subject to their owners' security
processes. A report that appears to belong elsewhere may be redirected without
exposing its contents publicly.

This source tree targets v1.2.0. Report issues against the exact
version or commit you use; support scope is recorded in published release notes.

## Trusted inputs

Generic Hugging Face loaders and Pi0.5 do not execute remote model code.
RhinoVLA `runtime_factory` values are executable installed integrations and
must come from reviewed configuration. Hy-Embodied `norm_stats.pkl` files are
also executable inputs: the required SHA-256 pins reviewed bytes but does not
make an untrusted pickle safe.

Treat model checkpoints, tokenizers, processors, configuration, and
pickle-backed statistics as trusted inputs. Review their source, revision,
license, digest, and configuration before loading them.

RhinoForge declares `torch==2.10.0`. Review upstream security advisories and
platform-provider guidance for the installed dependency versions and application
code paths. Reassess dependencies when adding model loaders or execution APIs.

Privileged board-device access is intended only for trusted users on a
single-tenant host. Do not expose it to untrusted code or mutually untrusted
tenants.
