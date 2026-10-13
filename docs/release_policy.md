# Version and release policy

RhinoForge, Runtime and Rhino Launch share a distribution version. The root
[VERSION](../VERSION) is the source for Python metadata and the numeric CMake
version. The release suffix remains part of Python metadata and package names.

| Line | Version | Purpose |
|---|---|---|
| `main` | `1.3.0.dev0` | Next release development |
| `release/1.2` | `1.2.0rc1`, then `1.2.0` | Release candidates and stabilization |
| `release/1.2` after release | `1.2.1` | Compatible fixes |
| Historical tags | Original version | Reproducible source and downloads |

Cut the release branch before advancing `main`. The Launch development line
uses `develop` with the same version stages. Release branches accept reviewed
fixes; development continues on the next version. A version's presence in the
archive does not by itself define whether it receives maintenance.

Tags use `v` followed by the complete version, such as `v1.2.0rc1`. Publish
candidate, stable and development entries separately. A development entry may
provide source only; it must not imply that matching runtime binaries exist.
The current release candidate is `1.2.0rc1`. Keep it in the current-candidate
section while release preparation continues; advancing development to `1.3`
does not archive it. Pre-publication source updates stay on `release/1.2`.
Keep existing package bytes, tags and checksums; identify replacement builds
by source commit in separate download paths. After public release, a new
candidate receives the next `rc` number and a stable fix the next patch version.
Archive a candidate when it is superseded, retaining its dates, component
commits and checksums.

## Build and dependency versions

[runtime-dependencies.json](../release/runtime-dependencies.json) records the
Launch source commit, complete `release_version`, numeric `cmake_version` and
`abi_version`. Its legacy `version` field equals `cmake_version`. For `1.2.0rc1`,
CMake uses `1.2.0` and also checks the full candidate version. The library SONAME
tracks ABI compatibility independently. Python's `API_VERSION` is also a separate
compatibility level and is unchanged by a distribution release.

Runtime contains exactly the main and DDR-expansion operator assets pinned by
commit and checksum. Their component source versions need not match the bundle
version. Keep each payload and its `.kernels` manifest together.

Run `python scripts/check_release_version.py` before packaging; CI also checks
the branch or tag against `VERSION`. Validate Python metadata, the selected Launch
package and the affected build or runtime path before publishing. Keep model
evaluation records outside release artifacts.

## 中文说明

RhinoForge、Runtime 和 Rhino Launch 使用同一发行版本，`VERSION` 为唯一版本源。
从 `main` 切出 `release/1.2` 后，候选版使用 `1.2.0rc1`，开发线进入 `1.3.0.dev0`；
Launch 的开发分支仍为 `develop`。正式版使用 `1.2.0`，兼容修复使用 `1.2.1`。

下载页分别显示开发版、候选版、稳定版和历史版本。开发版可仅提供源码；归档不等于
停止维护。当前待发布版本为 `1.2.0rc1`，准备期间保持当前候选入口，源码更新进入
`release/1.2`；开发线进入 `1.3` 不触发候选版归档。原标签、包和校验和保持不变，
重建包按源码 commit 使用独立下载路径。公开发布后的新候选递增 `rc`，正式版修复
递增补丁号；候选被替代后再归档，保留日期、组件 commit 和校验和。

CMake 数值版本不含 `rc` 或 `.dev` 后缀，构建时另外核对完整发行版本。Launch ABI
和 Python API 兼容版本独立维护。算子资产仍固定来自 main 与 DDR-expansion 两个
分支，依赖清单记录准确 commit 和校验和。
