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
Existing tags, package bytes and checksum files remain immutable. A changed
candidate receives the next `rc` number; a changed stable release receives the
next patch version. Keep old downloads, dates, component commits and checksums.

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
停止维护。已发布的标签和文件保持不变，新候选递增 `rc`，正式版修复递增补丁号。
保留原包、日期、组件 commit 和校验和。

CMake 数值版本不含 `rc` 或 `.dev` 后缀，构建时另外核对完整发行版本。Launch ABI
和 Python API 兼容版本独立维护。算子资产仍固定来自 main 与 DDR-expansion 两个
分支，依赖清单记录准确 commit 和校验和。
