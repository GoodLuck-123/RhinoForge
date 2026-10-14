# Version and release policy

RhinoForge, Runtime and Rhino Launch share a distribution version. The root
[VERSION](../VERSION) supplies Python metadata and the numeric CMake version.
Package names retain the full version, including any development suffix.

| Line | Version | Purpose |
|---|---|---|
| `main` | `1.3.0.dev0` | Next release development |
| `release/1.2` | `1.2.0` | Current stable release |
| `release/1.2` after release | `1.2.1` | Compatible fixes |
| Historical tags | Original version | Reproducible source and downloads |

Cut the release branch before advancing `main`. Launch uses `develop` for its
development line and follows the same version stages. Merge release fixes back
into development while keeping its next-version metadata.

Tag the stable release `v1.2.0` and retain a dated archive tag with its source
commit. The current download entry is `1.2.0`; superseded candidates remain in
history. Advancing development does not retire the current stable release.
A development entry may provide source only. Preserve published tags, package
bytes and checksums; keep build dates and component commits with each archive.
New stable fixes increment the patch version. Archive status does not determine
whether a release receives maintenance.

## Build and dependency versions

[runtime-dependencies.json](../release/runtime-dependencies.json) records the
Launch source commit, full `release_version`, numeric `cmake_version` and
`abi_version`. The legacy `version` field equals `cmake_version`. CMake checks
both numeric and full versions. Library SONAME and Python `API_VERSION` track
compatibility independently of distribution versions.

Runtime contains exactly the main and DDR-expansion operator assets pinned by
commit and checksum. Their component source versions need not match the bundle
version. Keep each payload and its `.kernels` manifest together.

Run `python scripts/check_release_version.py` before packaging; CI also checks
the branch or tag against `VERSION`. Validate Python metadata, the selected Launch
package and the affected build or runtime path before publishing. Keep model
evaluation records outside release artifacts.

## 中文说明

RhinoForge、Runtime 和 Rhino Launch 使用同一发行版本，`VERSION` 为版本源。
`release/1.2` 发布正式版 `1.2.0`，`main` 开发 `1.3.0.dev0`；Launch 开发分支为
`develop`。兼容修复递增补丁号，并合回开发分支，保留开发线的版本信息。

正式版使用 `v1.2.0` 标签，同时保留含日期与源码 commit 的归档标签。下载页当前
版本为 `1.2.0`，被替代的候选版进入历史记录。开发线升版不影响当前正式版入口。
开发版可仅提供源码。已发布的标签、包和校验和保持不变，归档记录日期和组件
commit；归档状态不代表停止维护。

CMake 数值版本不含开发后缀，构建时另行核对完整发行版本。Launch ABI 和 Python
API 兼容版本独立维护。算子资产固定来自 main 与 DDR-expansion 两个分支，依赖
清单记录准确 commit 和校验和。发布前检查版本、包装和受影响的构建或运行路径；
模型评测记录不进入发行资产。
