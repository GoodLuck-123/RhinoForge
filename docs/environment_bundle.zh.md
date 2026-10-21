# RhinoForge 依赖环境包

简体中文 | [English](environment_bundle.md)

当前环境版本为 [`deps-2026.10.1`（预览）](https://github.com/HUIXI-AI/RhinoForge/releases/tag/deps-2026.10.1)，
面向 Linux aarch64、Python 3.12。用户部署沿用[统一安装指南](getting_started.zh.md#bundle-deploy)，
本页说明包的范围、源码关联和维护流程，不另设一套开发环境。

## 四部分独立交付

| 部分 | 内容 | 交付方式 |
|---|---|---|
| 平台基础环境 | 板卡 OS、内核、驱动、完整 SDK（含用户态库）及权限配置 | 平台方提供和维护 |
| 依赖环境包 | Python、第三方运行依赖及构建、开发和音视频工具 | GitHub Releases |
| RhinoForge | Git 历史、源码、示例、文档；在目标机编译的 native extension | 从 GitHub 检出指定 tag/commit，本地构建 |
| 运行资产 | Rhino Launch 开发包、main / DDR-expansion 算子及相邻清单 | 按提供方条款单独交付 |

模型权重、模型输入和模型专用的外部源码仍按各模型指南单独准备。
公开依赖包不包含 RhinoForge 源码、wheel、已安装的 `rhinoforge` / `rpu_backend`
或 `rpu_backend.so`；也不包含任何 SDK 库、Launch、算子、模型或私人缓存。
`libHxMem.so`、`libHxil.so`、`librpu_api.so` 均属于板卡 SDK，由平台方提供。

## 推荐的平台版本

当前推荐并已验证的板卡 SDK 为 **`EA_R1SDK_2026_08F-2156007a`**。
已有验证基线为 R1、Linux aarch64、Ubuntu 22.04、glibc 2.35、GCC 11.4，
应用解释器为 Python 3.12.15。推荐基线及平台准备步骤统一见[安装指南](getting_started.zh.md#前置条件由平台方提供)。
具体 OS / 内核 / 驱动与 SDK 的配套关系、安装方法和权限由平台方确认。
其他 SDK 版本需确认适配并验证，推荐版本不是通用最低版本。

**Rhino Launch 1.2.0 是另一个组件的版本**，当前 ABI 为 1，提交为
`3dd1b410f5f46373997fef3dfbf9b67eecec6a84`。Launch 和两份算子资产继续按
[runtime-dependencies.json](../release/runtime-dependencies.json)精确匹配。
环境包版本、RhinoForge 版本、板卡 SDK 版本和 Launch 版本彼此独立。

## 包内范围

依赖包采用一个 `full` 配置，覆盖默认依赖、`vision`、`vla`、`audio` 和 `dev`，
不分别维护入门包与开发包。依赖声明来自选定源码的
[pyproject.toml](../pyproject.toml)；最终发布清单必须固定全部直接和传递依赖。

| 类别 | 内容 |
|---|---|
| Python 环境 | Python 3.12、开发头文件、pip、兼容的 packaging / setuptools / wheel、`conda-unpack` |
| 推理依赖 | Torch、Transformers、NumPy、Accelerate、Hugging Face Hub 等默认依赖及完整传递依赖 |
| 扩展依赖 | torchvision、torchaudio、LeRobot、Diffusers、图像与音频 Python 库等所选 extras |
| 构建与开发 | scikit-build-core、CMake、Ninja、build、pytest、conda-pack |
| 音视频工具 | SoX CLI、FFmpeg、libsndfile 及实际构建所需的运行库 |
| 可复现和分发材料 | 包清单、精确锁文件、来源与校验和、许可证和必要的第三方源码材料 |

保留 conda-pack 所需的前缀迁移记录，避免把它们当作缓存删除。制作包使用
Miniconda3；交付物只包含应用环境，不包含 Miniconda base 或 conda 管理器。
目标机仍需 Bash、Git、下载/解包工具及系统 C/C++ 构建工具，包括 GCC/G++、
binutils、glibc 开发头和 pkg-config。独立安装器使用宿主 Python 3.10 或更新版本；
运行/编译 RF 使用包内 Python 3.12。系统工具的安装沿用统一安装指南。

这些依赖会低频更新，但安全修复、Python/Torch ABI 或模型依赖变化仍需要新包。
当前源码中的固定版本是配方起点，不能直接当作新公开包已完成的兼容性或安全验证。

## 用户部署与二次开发

两条路径汇合到相同的源码构建、模型准备、运行和开发步骤：

1. 路径一：安装 Miniconda3 → 创建 `RHINOFORGE_ENV` → 安装第三方依赖。
2. 路径二：下载依赖包 → SHA256 校验 → 解包并迁移前缀 → 激活 `RHINOFORGE_ENV`。
3. 获取指定 GitHub 源码，配置平台 SDK 和单独交付的运行资产，核对源码依赖后编译 RF。
4. 准备模型和输入，按平台设备使用约定运行；二次开发继续使用同一环境。

路径二的“一键”只覆盖依赖环境部署。它省去 Torch 等依赖的逐个下载和安装，
**仍需编译 RhinoForge**。离线部署还需提前取得源码、系统构建工具、SDK、运行资产
及模型；一个依赖压缩包不能完成全部准备。

用户从指定 tag/commit 开始，在 Git 仓库创建自己的开发分支；源码目录与依赖
环境目录分开。初次编译和后续 editable 安装共用一个 `RHINOFORGE_ENV`。
更新 Python/C++ 代码后重新安装/构建，不能只更新 Python 文件而保留不匹配的 native。
普通源码更新若未改变规范化依赖声明和 ABI，可继续使用原环境。当前工具要求依赖
声明变化时制作并验证新的环境版本，即使旧包中的已安装版本仍满足新约束；
不能只改源码关联记录，也不隐式升级环境中的 Torch 等依赖。

工具使用的部署目录：

    rhinoforge-work/
      dependencies/
        activate.sh
        environment-manifest.json
        locks/, licenses/, sources/    # 第三方配方、许可证及必要源码材料
        env/                          # RHINOFORGE_ENV；初始不含 RF
      source/                         # RHINOFORGE_SOURCE；用户的 Git clone
      build/                          # RHINOFORGE_BUILD
      runtime/                        # 单独交付的 Launch 和算子
      models/, results/               # 用户另行准备/生成

SDK 可安装在平台指定的其他位置。激活脚本只设置 Python 环境和环境身份，不指向
不存在的内置源码/运行资产，也不覆盖平台 SDK 的动态库路径。先激活环境，再按
[运行资产说明](runtime_assets.zh.md)配置 Launch、算子和 SDK。
直接部署到最终路径；迁移后更换目录时从原压缩包重新部署。架构和 glibc 检查
仅是必要条件，不代表任意 Linux 或 SDK 组合受支持。

## 与 GitHub 源码的版本关联

环境独立编号为 `deps-2026.10.1`，归档文件为
`rhinoforge-deps-2026.10.1-full-linux-aarch64-py312.tar.gz`。
已发布文件和校验和保持不变，不使用可变的 latest 地址作为复现依据。
多个 RF 提交可引用同一个环境。

仓库使用 [`release/environment.json`](../release/environment.json) 记录当前源码推荐的环境，
包含实际下载信息。字段如下：

| 字段 | 作用 |
|---|---|
| `schema`、`environment_id`、`profile` | 新依赖包格式、独立环境版本和 extras 范围 |
| `platform`、`python` | 架构、验证过的 OS/glibc 基线和 Python ABI |
| `requirements_sha256` | 默认/extras/构建依赖及安全约束的规范化摘要，不包含 RF 自身版本号 |
| `archive_url`、`archive_sha256` | 真实发布文件及其 SHA256 |
| `manifest_sha256`、`lock_sha256` | 绑定随包清单与精确依赖锁 |

包内 `environment-manifest.json` 使用新的 `rhinoforge-dependency-env/v1` `schema`，
记录环境身份、构建工具版本、精确 Conda/pip 包版本、构建号、来源、下载文件摘要
及 `recipe_source_commit`。该 commit 说明配方来源，不表示内含 RF，也不限制环境
只能用于一个提交。SDK 推荐和 Launch/算子身份由源码文档及运行资产锁维护，
不能把这些平台组件写入包内库清单。

`scripts/dependency_environment.py` 在构建前比对环境身份和摘要，核对源码的 Python、运行和构建
依赖，包括 extras 和环境标记。`pip check` 无法检查尚未安装的 RF 的需求；
这些检查通过后才使用 `--no-build-isolation` `--no-deps` 构建，避免重下或更改依赖。
依赖摘要属于不可变的归档清单。需求变更时，即使旧包中的版本仍可满足，
也须制作新的归档和环境版本、重新验证并更新源码关联记录。

## 制作和发布流程

1. 用 Miniconda3 创建全新依赖环境，先安装 Conda 包，再安装锁定的 pip 包。
   不安装 RF，不复制现有部署目录；保持 Conda 管理的文件未被 pip 覆盖。
2. 检查清单和文件，拒绝 RF 安装残留、editable 路径、SDK、Launch/算子、模型、
   凭据和私人配置。为每个实际第三方构建收集许可证、声明及必要源码；
   SoX 的材料不能代替 FFmpeg、libsndfile 等组件的材料。
3. 用 `pack_dependency_env.py` 打包，在另一绝对路径部署，验证迁移、依赖、导入和音视频工具。
   `install_dependency_env.py` 保留 SHA256、平台、路径安全检查及失败回滚，明确显示“依赖环境
   已安装，RhinoForge 尚未构建”。
4. 用部署后的环境从指定 GitHub 提交构建 RF，配合推荐 SDK 和独立运行资产
   跑通 Qwen3-0.6B FP16；受依赖变化影响的 vision/vla/audio 路径按
   [验证策略](validation_policy.zh.md)检查。导入成功不等于模型验证通过。
   构建/运行结果不能写回待发布的依赖包。
5. 在 GitHub Releases 提供归档、SHA256、manifest、精确锁、许可证及必要源码，
   更新源码环境关联和统一指南。环境 Release 使用独立 `deps-*` 标签；
   `check_release_version.py` 对这些标签调用 `check_dependency_release.py`，核对
   环境身份、配方和下载摘要。原有 `v*` 源码标签与 `VERSION` 的检查保持不变。

公开 GitHub Releases 是依赖包的统一入口，SDK 和受控运行资产仍沿各自渠道交付。
环境预览版不替代 RhinoForge 的稳定源码 Release，也不修改其 latest 入口。

## 现有工具与历史包

`scripts/pack_conda_bundle.py` 和 `scripts/install_conda_bundle.py` 目前服务于旧
`rhinoforge-conda-bundle/v1` 完整包，包含已安装 RF、源码、Launch 和算子，并支持
可选 SDK 库。它们继续只处理历史格式。新的
`pack_dependency_env.py` / `install_dependency_env.py` 使用独立 schema，拒绝将旧完整包
作为纯依赖包安装；旧工具不作为公开依赖包流程的入口。

内部完整归档不更名为公开依赖包，也不通过删除几个目录后直接发布。
纯依赖包不修改 SDK、Launch 或算子资产；它是独立制作的第三方依赖环境制品。

## 工具使用与发行检查

维护人员在未安装 RF 的干净依赖环境中准备全部第三方库，并在环境外准备
`licenses/`、`sources/`、`locks/`。自编 SoX 的安装文件用
`locks/native-packages.json` 记录版本、公共来源、源码 SHA256 和逐文件 SHA256。
Conda post-link 生成的加载器索引也须验证生成工具及引用库的包归属、重新生成比对，
再按精确文件摘要登记；不能把任意未知文件加入清单来绕过检查。
打包器会拒绝未登记的文件。普通部署用户无需执行此制作步骤。

```bash
python scripts/dependency_environment.py --source "$RHINOFORGE_SOURCE" --print-requirements
python scripts/pack_dependency_env.py \
  --env "$RHINOFORGE_CLEAN_DEPENDENCIES" --source "$RHINOFORGE_SOURCE" \
  --redistribution "$RHINOFORGE_REDISTRIBUTION" \
  --environment-id deps-2026.10.1 \
  --output "$RHINOFORGE_DIST/rhinoforge-deps-2026.10.1-full-linux-aarch64-py312.tar.gz"
```

以上变量分别指干净依赖前缀、源码、经审核的第三方材料目录和已存在的输出目录。
制作过程需要保留原始 Conda 包缓存；公开归档不包含包缓存或 Miniconda 管理器。
发布前用真实归档与提取出的 manifest 检查源码绑定：

```bash
python scripts/check_dependency_release.py --tag deps-2026.10.1 \
  --archive "$RHINOFORGE_DIST/rhinoforge-deps-2026.10.1-full-linux-aarch64-py312.tar.gz" \
  --manifest "$RHINOFORGE_DIST/environment-manifest.json"
```

SHA256、来源和许可证检查不代表所有第三方库没有漏洞。固定的模型依赖可能保留
上游已知问题，具体包版本、触发条件和使用限制随包记录在 `KNOWN_ISSUES.md`。
使用经确认的模型和输入，不把该环境当作执行任意外来模型代码或文件的隔离沙箱。
