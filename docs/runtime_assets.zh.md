# 外部运行时资产

简体中文 | [English](runtime_assets.md)

RhinoForge 源码和 wheel 不包含 Rhino Launch 二进制、主算子资产与扩展算子资产、板卡 SDK 或
模型权重。请向运行时提供方获取与待安装源码版本匹配的资产，按随包条款和完整性说明
使用。[完整 Conda 环境包](getting_started.zh.md#6-部署完整环境包)可以按提供方条款
随包包含这些用户态运行时资产；实际内容和仍需外部提供的 SDK 组件以包内清单为准。

当前 `1.3.0.dev0` 开发线依赖已归档的 **Rhino Launch 1.2.0**，
commit 为 `3dd1b410f5f46373997fef3dfbf9b67eecec6a84`，CMake 数值版本为 `1.2.0`，ABI 为 `1`。
向提供方获取正式版 `RhinoForge-runtime-v1.2.0.tar.gz` 及其 `.tar.gz.sha256`。
其中 Launch SDK 目录为 `rhino-launch-kernel-v1.2.0-linux-aarch64`，main 和
DDR-expansion 资产与[依赖清单](../release/runtime-dependencies.json)一致，
两份资产都需保留相邻的 `.ref.kernels` 清单。
历史候选包可能使用同一整包文件名，须核对校验和、`RELEASE.txt` 和 Launch commit。
`v1.1.0-2-gbdc0dae` 目录对应旧 SDK，其 CMake 版本为 `1.0.0`，不满足当前依赖。
见[发布流程](release_policy.md)。

按当前 checkout 的依赖清单选择资产。提供方的开发版下载页或包内 README
可能描述旧源码：`1.3.0.dev0` 不意味着依赖 Launch `1.3.0.dev0`。
本源码线使用上面标识的稳定 1.2.0 资产；页面不一致时向提供方申请修正文案，
不要绕过 CMake 或资产身份检查。

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

从提供方的受控交付入口下载 `RhinoForge-runtime-v1.2.0.tar.gz` 和其 `.sha256`，
放入同一下载目录。以下命令校验并解开两层包；只解外层包并不会产生可安装的
Launch 前缀。使用源码和 Python 环境之外的新目录，任何校验失败都应停止：

```bash
export RHINOFORGE_DOWNLOADS="/absolute/path/to/runtime-downloads"
export RHINOFORGE_RUNTIME="/absolute/path/to/new/runtime"
cd "$RHINOFORGE_DOWNLOADS"
sha256sum -c RhinoForge-runtime-v1.2.0.tar.gz.sha256
mkdir -p "$RHINOFORGE_RUNTIME"
tar -xzf RhinoForge-runtime-v1.2.0.tar.gz -C "$RHINOFORGE_RUNTIME"
export RHINOFORGE_BUNDLE="$RHINOFORGE_RUNTIME/RhinoForge-runtime-v1.2.0"
cd "$RHINOFORGE_BUNDLE"
sha256sum -c SHA256SUMS
cat RELEASE.txt
tar -xzf rhino-launch-kernel-v1.2.0-linux-aarch64.tar.gz -C "$RHINOFORGE_RUNTIME"

export RHINO_LAUNCH_ROOT="$RHINOFORGE_RUNTIME/rhino-launch-kernel-v1.2.0-linux-aarch64"
export RPU_KERNEL_LIB_PATH="$RHINOFORGE_BUNDLE/rhinoOpLib_current.ref"
export RPU_KERNEL_LIB_PATH_EXPANSION="$RHINOFORGE_BUNDLE/rhinoExpansionOpLib_current.ref"
export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION.kernels"
ls "$RHINO_LAUNCH_ROOT/lib/"librhino_launch.so*
```

解包结果为：

```text
runtime/
  RhinoForge-runtime-v1.2.0/
    RELEASE.txt, SHA256SUMS
    rhino-launch-kernel-v1.2.0-linux-aarch64.tar.gz
    rhinoOpLib_current.ref, rhinoOpLib_current.ref.kernels
    rhinoExpansionOpLib_current.ref, rhinoExpansionOpLib_current.ref.kernels
  rhino-launch-kernel-v1.2.0-linux-aarch64/
    include/
    lib/                       # 动态库及 cmake/rhino_launch/
```

将 `RELEASE.txt` 与[源码依赖清单](../release/runtime-dependencies.json)对照：
Launch 的 commit、版本、ABI 以及两份算子资产的 commit、SHA256 均须匹配。
随包校验和验证下载完整性，源码依赖清单确定该包是否适配当前源码。
清除继承环境中冲突的 Launch 路径，同时保留 SDK
必需路径。凭据和本机路径放在 shell 环境中，不写入模型 TOML。随后参考
[统一安装、运行与开发指南](getting_started.zh.md)。

## 板卡 SDK 与设备权限

Runtime 包**不安装**板卡 SDK 或内核驱动，也没有绑定可跨平台使用的 SDK 最低版本。
请通过提供方交付渠道获取板卡 SDK/driver 包、支持的 OS/内核/版本组合、安装命令、
动态库配置及权限和设备锁规范。本仓库没有通用 SDK 下载地址或驱动安装命令；
裸板应先完成平台初始化，再构建 RhinoForge。记录实际 SDK/driver 版本；
保留预装板卡基础环境，只能称为独立应用安装，不能称为 SDK/driver 从零安装。

配置平台和新 Launch 路径后检查：

```bash
uname -smr
ldd "$RHINO_LAUNCH_ROOT/lib/librhino_launch.so"
ls -l /dev/rpu* /dev/mem
```

`ldd` 中依赖必须全部解析到所选 Launch 或平台 SDK/系统库；出现 `not found`
时按 SDK 文档配置 loader 路径。若包内没有无版本号的 `.so` 链接，使用实际库文件名。
节点存在不等于有权限：运行进程需要对至少一个 `/dev/rpu*` 节点和 `/dev/mem`
执行 `O_RDWR` 打开。用即将执行推理的账号检查，以下代码不读写设备内容：

```bash
python3 - <<'PY'
import glob, os
candidates = sorted(glob.glob("/dev/rpu*"))
if not candidates:
    raise SystemExit("No /dev/rpu* node: check the platform driver")
for path in candidates:
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except OSError as error:
        print(f"{path}: {error}")
    else:
        os.close(fd)
        print(f"O_RDWR available: {path}")
        break
else:
    raise SystemExit("No accessible RPU node")
fd = os.open("/dev/mem", os.O_RDWR | os.O_CLOEXEC)
os.close(fd)
print("O_RDWR available: /dev/mem")
PY
```

权限不足时由平台管理员设置受支持的访问策略，或使用获准的特权启动器。
不要全局放宽 `/dev/mem` 权限。权限修复后启动新 Python 进程；普通 CPU-only
导入可在 native backend 不可用时成功，`--check-config` 也不检查设备。
空闲检查、锁及特权运行示例见[入门指南](getting_started.zh.md#3-在空闲设备上运行)。

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
| CMake 报 Launch 版本不匹配 | 按依赖清单选择正式归档 SDK，同时更新构建和运行路径；不要改写 SDK 版本文件或移除 EXACT 检查 |
| CMake 拒绝缺失的能力 | 获取匹配的 sanitized Release Launch 包 |
| Launch 动态库无法加载 | 检查其 `lib` 路径及板卡 SDK 动态库路径 |
| 资产、清单或所需 kernel 缺失 | 获取配套资产和未修改的清单 |
| 设备节点缺失、`Permission denied` 或 `Operation not permitted` | 检查驱动，以及推理账号对 `/dev/rpu*` 和 `/dev/mem` 的 `O_RDWR` 权限；修复后重启 Python |
| 旧代码报 `PyTorch is not linked with support for rpu devices` | 先查看此前的 native loader 诊断和节点权限；可能是后端注册被跳过，而非 PyTorch 编译问题 |
