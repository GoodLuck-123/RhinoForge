# 外部运行时资产

简体中文 | [English](runtime_assets.md)

RhinoForge 源码和 wheel 不包含 Rhino Launch 二进制、主算子资产与扩展算子资产、板卡 SDK 或
模型权重。请向运行时提供方获取与待安装源码版本匹配的资产，按随包条款和完整性说明
使用。

RhinoForge v1.2.0 的整包名为 `RhinoForge-runtime-v1.2.0.tar.gz`，
请向提供方同时获取其 `.tar.gz.sha256`。解压目录为 `RhinoForge-runtime-v1.2.0/`，
其中包含 `rhinoOpLib_current.ref`、`rhinoExpansionOpLib_current.ref` 及各自相邻的 `.ref.kernels` 清单。
RhinoForge、Runtime、Rhino Launch 统一使用发行版本 `1.2.0`。
Launch 的 CMake 数值版本为 `1.2.0`，ABI 版本保持 `1`。
使用 `RELEASE.txt` 指定的准确 Launch 内包，版本规则见[发布流程](release_policy.md)。

## 申请运行时资产

可通过仓库 [GitHub Issues](https://github.com/HUIXI-AI/RhinoForge/issues/new) 提交资产申请，
注明源码版本或 commit、板卡平台，以及所需 Launch、main 和 DDR-expansion 组件。
公开 issue 中不附受限文件、凭据或本机信息；资产获取按提供方的分发条款办理。

## 必需组件

- **Rhino Launch 1.2.0**：sanitized Release 包，包含头文件、动态库和
  `lib/cmake/rhino_launch/rhino_launchConfig.cmake`。当前 CMake 检查
  `graph-arena-pool-v2`、`register-state-token`、Chrome hwperf 能力，且要求
  不暴露 development Program API。
- **两份算子资产**：主分支和 DDR-expansion 分支各自产出的资产，以及各自相邻的 `<asset>.kernels` 清单：资产需匹配当前源码，
  并包含所用公开模型和精度配置要求的 kernel。
- **匹配的板卡 SDK/runtime**：按平台安装说明配置。

构建和运行时使用同一套 Launch。
不要将旧算子资产与新清单混用，也不要在本地拼接算子资产。

## 配置路径

接收资产时按提供方说明核对校验值，再将其解压到源码和 Python 环境之外的固定目录。
设置安装后的 Launch 前缀及原始算子资产路径：

```bash
export RHINO_LAUNCH_ROOT="/absolute/path/to/matching/launch-install"
export RPU_KERNEL_LIB_PATH="/absolute/path/to/RhinoForge-runtime-v1.2.0/rhinoOpLib_current.ref"
export RPU_KERNEL_LIB_PATH_EXPANSION="/absolute/path/to/RhinoForge-runtime-v1.2.0/rhinoExpansionOpLib_current.ref"
export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION.kernels"
```

将示例路径替换为实际交付文件。清除继承环境中冲突的 Launch 路径，同时保留 SDK
必需路径。凭据和本机路径放在 shell 环境中，不写入模型 TOML。随后参考
[入门指南](getting_started.zh.md)或[源码构建指南](development_build.md)。

## 算子库角色与替换

两种角色固定为 main 和 DDR-expansion，由 [依赖清单](../release/runtime-dependencies.json)记录源码提交及 SHA256。源码入口和安装后的包使用相同角色，切换模型不增加模型专用 REF。

`RPU_KERNEL_LIB_PATH` 与 `RPU_KERNEL_LIB_PATH_EXPANSION` 可指向安装目录中的同内容副本。入口核对选中资产与清单，拒绝内容不符、角色互换或冲突的继承路径。GR00T 配置不接受 `example.expansion_ref`。替换任一库后启动新进程。

更新完整 main、expansion 资产时，同步更新相邻清单及源码、大小、SHA256 记录。
局部构建或本地拼接的 payload 不能替代任何一个角色。新增算子实现归算子仓库，
本仓库维护模型路由和 host wrapper。复现旧源码时，应配套使用该版本归档的 runtime 和原始
校验和；不能将当前源码指向旧 payload 却保留当前依赖记录。

## 算子清单

`.kernels` 必须与资产相邻。首行为 `rhinoforge-kernels-v1`，第二行为
`asset-size=<bytes>`，后续为排序且无重复的 ASCII kernel 名称。运行时在解析 kernel
前检查格式和资产大小。这能发现元数据缺失及大小不匹配，但不能验证算子实现，也不
替代提供方的完整性检查。

缺少所需 kernel 时，向提供方获取匹配的公开模型算子资产。给清单加一个名称不会给
资产增加实现。算子源码和资产内容不在本仓库中分发。

## 常见配置问题

| 现象 | 处理 |
|---|---|
| CMake 找不到 `rhino_launch` | 设置 Launch 前缀，或传入 `-Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch"` |
| CMake 拒绝缺失的能力 | 获取匹配的 sanitized Release Launch 包 |
| Launch 动态库无法加载 | 检查其 `lib` 路径及板卡 SDK 动态库路径 |
| 资产、清单或所需 kernel 缺失 | 获取配套资产和未修改的清单 |
| 设备不可用 | 检查设备权限和平台运行时配置 |
