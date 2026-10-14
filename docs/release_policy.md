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

Tag each stable release once, such as `v1.2.0`. Its Release records the date,
source and component commits, and package checksums; a duplicate `archive/` tag
is unnecessary. The current download entry is `1.2.0`. Keep superseded candidate
packages and their original checksums in the download history. Before removing
a candidate tag or Release, migrate its records and update download links.

Keep stable tags and published package bytes fixed. Advance `main` independently;
compatible stable fixes receive a new patch version. Retain `main` and the active
release branches, and remove merged working branches after checking for new
commits. A development entry may provide source only.

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

每个正式版只保留一个标准标签，如 `v1.2.0`；Release 记录日期、源码及组件 commit
和包校验和，无需重复创建 `archive/` 标签。当前下载入口为 `1.2.0`。被替代的候选包
及原校验和保留在下载历史中，删除候选标签或 Release 前先迁移记录并更新链接。

正式标签和已发布包保持固定。`main` 独立开发，兼容修复递增补丁号。长期保留
`main` 和仍在维护的 release 分支；工作分支合并后确认没有新增提交即可删除。
开发版可仅提供源码。

CMake 数值版本不含开发后缀，构建时另行核对完整发行版本。Launch ABI 和 Python
API 兼容版本独立维护。算子资产固定来自 main 与 DDR-expansion 两个分支，依赖
清单记录准确 commit 和校验和。发布前检查版本、包装和受影响的构建或运行路径；
模型评测记录不进入发行资产。
