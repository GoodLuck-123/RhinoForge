# RhinoForge 安装、运行与开发

简体中文 | [English](getting_started.md)

本指南从独立的 RhinoForge 应用环境开始，最终跑通 Qwen3-0.6B FP16；首次部署和
后续开发共用一个 `RHINOFORGE_ENV`。开始前，必须先取得平台方提供的板卡基础环境
和配套运行资产。

## 前置条件：由平台方提供

以下组件及其适配关系必须由**平台方提供并确认**，不能通过安装 RhinoForge 或
Conda 环境替代：

| 平台交付项 | 开始部署前须具备的内容 |
|---|---|
| 板卡 OS / 内核 | 与板卡匹配的系统镜像、版本和初始化说明 |
| 板卡驱动 | 与 OS / 内核匹配的驱动，设备节点及推理账号访问权限 |
| 板卡 SDK | 匹配的 SDK/runtime、安装说明、必需的用户态库路径 |
| RhinoForge 运行资产 | 与所选源码版本匹配的 Rhino Launch、main / DDR-expansion 算子资产、各自 `.kernels` 清单、SHA256 和分发条款 |
| 运行约定 | 支持的 OS / 驱动 / SDK 组合，以及空闲检查、设备锁或调度器的使用方式 |

请先按平台交付说明完成系统、驱动和 SDK 的配置，并取得匹配的运行资产；具体资产
身份和检查方法见[运行时资产说明](runtime_assets.zh.md)。本指南中的“从零构建”指
新建应用环境并构建 RhinoForge，不包含裸板刷机、驱动安装或 SDK 适配。
推荐板卡 SDK 为 **`EA_R1SDK_2026_08F-2156007a`**；已有验证基线为
R1 Linux aarch64、Ubuntu 22.04、glibc 2.35、GCC 11.4 和 Python 3.12.15。
这是已验证的组合，不是通用最低版本；其他 SDK 需由平台方确认适配并验证。
**SDK 用户态库也由平台方提供，不进入我们的依赖环境包。**
Rhino Launch **1.2.0** 是独立运行组件，不能与板卡 SDK 版本混用。

## 选择一条部署路径

| 路径 | 需要做什么 | 适用情况 |
|---|---|---|
| [路径一：从 Miniconda3 开始构建](#source-build) | 安装 Miniconda3 → 创建 Python 3.12 环境 → 安装依赖 → 获取源码和运行资产并编译 | 逐步安装依赖，也适合自定义环境 |
| [路径二：使用我们提供的依赖环境包](#bundle-deploy) | 下载、校验、解包环境 → 从 GitHub 获取源码 → 配置运行资产并编译 | 省去 Torch 等第三方依赖的逐个下载与安装 |

两条路径都使用一个 `RHINOFORGE_ENV`，都从 GitHub 获取并编译 RhinoForge，
随后共用模型准备、运行和开发步骤。路径二只替代依赖环境的创建与安装，不包含
RhinoForge、板卡 SDK、Launch 或算子，也不免除本地编译。
当前依赖环境版本为 [`deps-2026.10.1`（预览）](https://github.com/HUIXI-AI/RhinoForge/releases/tag/deps-2026.10.1)，
包含默认依赖及全部扩展。具体范围、已知限制和版本关联见
[依赖环境包说明](environment_bundle.zh.md)。

<a id="1-安装-miniconda3-并构建源码"></a>
<a id="source-build"></a>

## 路径一：从 Miniconda3 开始构建

### 1.1 下载安装 Miniconda3

以下命令使用 Bash，面向 Linux **aarch64** RPU 板卡。下载前用 `uname -m` 确认
架构，x86_64 安装器或环境包不能用于 aarch64。选择新的、当前用户可写的目录，
预留下载文件、环境、源码、编译产物和模型的空间。示例使用 `$HOME`，也可统一改到
可写的数据盘。Ubuntu/Debian 先安装主机工具：

```bash
set -e
sudo apt-get update
sudo apt-get install --yes build-essential pkg-config git curl ca-certificates bzip2 xz-utils

export RHINOFORGE_MINICONDA="$HOME/miniconda3-rhinoforge"
export RHINOFORGE_DOWNLOADS="$HOME/rhinoforge-downloads"
export RHINOFORGE_WORKDIR="$HOME/rhinoforge"
export RHINOFORGE_SOURCE="$RHINOFORGE_WORKDIR/source"
export RHINOFORGE_ENV="$RHINOFORGE_WORKDIR/env"
export RHINOFORGE_BUILD="$RHINOFORGE_WORKDIR/build"
test "$(uname -m)" = aarch64
mkdir -p "$RHINOFORGE_DOWNLOADS"
cd "$RHINOFORGE_DOWNLOADS"
curl --fail --location --retry 3 --continue-at - \
  https://repo.anaconda.com/miniconda/Miniconda3-py312_26.9.3-0-Linux-aarch64.sh \
  --output Miniconda3-py312_26.9.3-0-Linux-aarch64.sh
printf '%s  %s\n' a7631abb0af9f2b28bb2778db6229be982471a6032a16fc5817542ec0b2de91a \
  Miniconda3-py312_26.9.3-0-Linux-aarch64.sh | sha256sum -c -
bash Miniconda3-py312_26.9.3-0-Linux-aarch64.sh -b -p "$RHINOFORGE_MINICONDA"
source "$RHINOFORGE_MINICONDA/etc/profile.d/conda.sh"
conda --version
```

安装器和 SHA256 来自[Miniconda 官方归档](https://repo.anaconda.com/miniconda/)。
使用无人值守参数 `-b` 前请阅读安装器条款。安装仅使用独立目录，不替换系统 Python，
也不修改 shell 启动文件。打开新 shell 时，先重新 `source` 同一份 `conda.sh`。

### 1.2 创建唯一的应用环境

清除历史安装的覆盖变量，再创建和激活应用环境。`--override-channels` 明确选用
的包源，不继承用户的 channel 列表；Miniconda 提供管理器，下面通过社区
`conda-forge` channel 获取 Python。`CONDA_PKGS_DIRS` 将本次安装的包缓存放在
独立工作目录内，不修改全局 Conda 配置，也不使用或清理其他用户的缓存。需要
音频工具时，可在下方 `conda create` 命令末尾追加 `ffmpeg libsndfile`，或按
[音频工具附录](#build-bundle)向同一环境补装。aarch64 的 SoX 构建步骤也见该附录。

```bash
unset PYTHONPATH PYTHONHOME RHINO_LAUNCH_ROOT CMAKE_PREFIX_PATH LD_LIBRARY_PATH
while IFS= read -r name; do
  case "$name" in RPU_*|LKN_*) unset "$name" ;; esac
done < <(compgen -v)
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CONDA_PKGS_DIRS="$RHINOFORGE_WORKDIR/conda-pkgs"
mkdir -p "$CONDA_PKGS_DIRS"
conda create --yes --override-channels --channel conda-forge \
  --prefix "$RHINOFORGE_ENV" python=3.12.15 "pip>=23.1" \
  "packaging>=24.2,<26" "setuptools>=71,<81"
conda activate "$RHINOFORGE_ENV"
python - <<'PY'
import os, pathlib, ssl, sys, sysconfig
from importlib.metadata import version
from packaging.version import Version
assert sys.version_info[:2] == (3, 12), sys.version
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(os.environ["RHINOFORGE_ENV"]).resolve()
header = pathlib.Path(sysconfig.get_path("include")) / "Python.h"
assert header.is_file(), f"Missing Python development header: {header}"
print("Python:", sys.executable, sys.version)
assert Version(version("pip")) >= Version("23.1")
assert Version("24.2") <= Version(version("packaging")) < Version("26")
assert Version("71") <= Version(version("setuptools")) < Version("81")
print("pip:", version("pip"))
print("packaging:", version("packaging"))
print("setuptools:", version("setuptools"))
print("Headers:", header)
print("TLS:", ssl.OPENSSL_VERSION)
PY
```

不要把应用依赖装进 Conda 的 `base` 环境。Conda 环境不适用 venv 的
`sys.prefix != sys.base_prefix` 判断，因此上面直接校验实际环境目录。
始终使用 `python -m pip`，确保 pip 属于当前解释器。后文的 `--config-settings`
是构建参数，`-c` 则是依赖约束文件参数，两者作用不同。

先安装 Conda 包，再安装 pip 包。LeRobot 0.5.1 要求 `packaging>=24.2,<26`、
`setuptools>=71,<81`；基础文本环境也由 Conda 预先安装这两个兼容范围，方便后续
在同一环境补齐全部 extras。若先用 Conda 安装更高版本，再让 pip 降级，会覆盖
Conda 管理的文件，导致 `conda-pack` 失败。pip、packaging、setuptools 保持由 Conda
管理；这里选定的 pip 已支持所需构建参数，无需再用 pip 升级自身。
`pip check` 成功不能单独证明 Conda 文件未被覆盖。

### 1.3 获取源码并配置平台方提供的运行资产

```bash
git clone https://github.com/HUIXI-AI/RhinoForge.git "$RHINOFORGE_SOURCE"
cd "$RHINOFORGE_SOURCE"
git rev-parse HEAD
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
"$CC" --version
"$CXX" --version
printf '#include <optional>\nint main() { std::optional<int> x{1}; return *x - 1; }\n' \
  | "$CXX" -std=c++17 -x c++ -fsyntax-only -
```

阅读所检出版本自带的指南。使用提供方指定的源码版本时，先切换到该 revision，
再继续后面的步骤，并记录提交 ID。更换编译器后使用新的构建目录；Python 的默认
编译器配置并不能证明板卡上已安装该编译器。

按[运行时校验及双层解包](runtime_assets.zh.md#配置路径)设置 `RHINO_LAUNCH_ROOT`、
`RPU_KERNEL_LIB_PATH`、`RPU_KERNEL_LIB_PATH_EXPANSION` 以及所选 Launch/SDK 的动态库
路径。当前 `1.3.0.dev0` 源码使用[依赖清单](../release/runtime-dependencies.json)
固定的 sanitized **Release Launch 1.2.0**，不要按 RhinoForge 版本号选择同名开发包。
校验两份算子资产及各自相邻的 `.kernels` 文件，再完成提供方的
[SDK 与设备权限检查](runtime_assets.zh.md#板卡-sdk-与设备权限)。
Conda 环境不会安装板卡 SDK 或内核驱动。

### 1.4 安装依赖并构建 RhinoForge

在同一 shell 和源码根目录继续。这里编译并安装普通 wheel；后续开发仍使用同一环境。
当前 [pyproject.toml](../pyproject.toml)固定 PyTorch **2.10.0**、Transformers
**5.3.0**。首次安装保留依赖解析；`constraints/security.txt` 是安全版本约束，
不等于全部依赖的锁文件。

```bash
cd "$RHINOFORGE_SOURCE"
python -m pip install -c constraints/security.txt \
  "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install . -c constraints/security.txt --no-build-isolation \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
python -m pip check
python - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
assert version("transformers") == "5.3.0"
assert version("torch").split("+")[0] == "2.10.0"
print("RhinoForge:", version("rhinoforge"))
print("Python package:", find_spec("rpu_backend").origin)
PY
```

末尾检查不加载 native extension，也不初始化设备。两个编译进程用于限制内存开销。
视觉、VLA、语音场景分别将源码安装命令中的 `.` 改为 `'.[vision]'`、`'.[vla]'`、
`'.[audio]'`；`'.[vision,vla,audio]'` 安装全部三组可选依赖。Qwen3 文本 FP16 无需
这些可选依赖，语音所需输入见[语音指南](qwen3_audio.zh.md)。

下载过慢的用户可使用[路径二](#bundle-deploy)准备环境；
离线部署仍须提前备齐源码、系统工具、SDK、运行资产和模型。
若自行使用镜像或 wheel 缓存，包必须适配 Python 3.12 和板卡架构，并记录下载源；
不要靠更换固定的 Torch/Transformers 版本解决下载速度问题。
安装成功后，[可编辑开发和 wheel 构建](#development)继续使用这个环境。

路径一到此完成。直接进入[第 3 节](#model-setup)，无需执行路径二。

<a id="6-部署完整环境包"></a>
<a id="bundle-deploy"></a>

## 路径二：使用依赖 Conda 环境包

### 2.1 获取源码与匹配的环境版本

依赖包通过 GitHub Releases 单独版本化，包含 Python 3.12、默认依赖、`vision`、
`vla`、`audio`、`dev` 和构建/音视频工具。它不包含 RhinoForge、SDK 库、Launch、
算子或模型；用户在自己的 Git checkout 中编译和开发。

从 [GitHub Release](https://github.com/HUIXI-AI/RhinoForge/releases/tag/deps-2026.10.1) 获取
与当前源码的 [`release/environment.json`](../release/environment.json) 匹配的包。
首个版本为预览包，已知依赖限制随 Release 的 `KNOWN_ISSUES.md` 提供；
用于已确认来源的本地模型、媒体和数据。

目标机需要 Bash、系统 Python 3.10 或更新版本，以及 C/C++ 构建工具。无需安装
Miniconda 管理器，也无需执行 `conda create`。Ubuntu/Debian 先准备系统工具，
再获取源码；使用提供方指定的 tag/commit 时，在读取环境记录前切换到该版本：

```bash
set -e
sudo apt-get update
sudo apt-get install --yes build-essential pkg-config git curl ca-certificates bzip2 xz-utils python3
export RHINOFORGE_WORKDIR="$HOME/rhinoforge"
export RHINOFORGE_SOURCE="$RHINOFORGE_WORKDIR/source"
export RHINOFORGE_BUILD="$RHINOFORGE_WORKDIR/build"
export RHINOFORGE_DEPENDENCIES="$RHINOFORGE_WORKDIR/dependencies"
test "$(uname -m)" = aarch64
mkdir -p "$RHINOFORGE_WORKDIR"
git clone https://github.com/HUIXI-AI/RhinoForge.git "$RHINOFORGE_SOURCE"
cd "$RHINOFORGE_SOURCE"
git rev-parse HEAD
```

`release/environment.json` 绑定当前源码与环境 ID、平台、依赖摘要、下载 URL 和
SHA256。环境版本独立于 RhinoForge；只改代码而不改依赖时，可以复用同一个包。

### 2.2 下载、校验和部署依赖环境

保持上一节的同一 shell。下面从当前 checkout 读取真实下载地址和校验和，由
安装器下载、校验、解包、执行前缀迁移并检查包依赖。安装脚本使用系统 Python，
解包后的 Python 3.12 用于构建和推理：

```bash
cd "$RHINOFORGE_SOURCE"
mapfile -t rf_bundle_fields < <(python3 - <<'PYENV'
import json
from pathlib import Path
record = json.loads(Path("release/environment.json").read_text())
print(record["archive_url"])
print(record["archive_sha256"])
PYENV
)
python3 scripts/install_dependency_env.py \
  --archive "${rf_bundle_fields[0]}" \
  --sha256 "${rf_bundle_fields[1]}" \
  --prefix "$RHINOFORGE_DEPENDENCIES"
unset rf_bundle_fields
source "$RHINOFORGE_DEPENDENCIES/activate.sh"
python scripts/dependency_environment.py \
  --source "$RHINOFORGE_SOURCE" --prefix "$RHINOFORGE_ENV"
```

安装目标目录必须尚不存在；父目录须存在且可写。离线板卡先传输归档和源码，
将 `--archive` 改为本地文件路径，SHA256 仍来自同一源码记录。安装过程不需要
下载 pip/Conda 依赖，不安装 RhinoForge，也不访问 RPU。

激活脚本设置 `RHINOFORGE_ENV="$RHINOFORGE_DEPENDENCIES/env"`，清除旧 Python/native
覆盖变量，并保留平台库路径。每次打开新 shell 均重新设置源码/工作目录并激活。
安装器检查归档身份、架构和 glibc；SDK、驱动和设备权限仍按平台说明确认。
直接部署到最终目录；迁移后换位置时，从原归档重新安装。

### 2.3 配置独立运行资产并编译 RhinoForge

按[运行资产说明](runtime_assets.zh.md#配置路径)设置 `RHINO_LAUNCH_ROOT`、
`RPU_KERNEL_LIB_PATH`、`RPU_KERNEL_LIB_PATH_EXPANSION` 和平台 SDK 动态库路径，
并完成[SDK 与权限检查](runtime_assets.zh.md#板卡-sdk-与设备权限)。
依赖环境不会提供或替换这些组件。

兼容性检查通过后，在同一 shell 构建 RF。`--no-index --no-deps --no-build-isolation`
使本次构建使用已核对的包内依赖，不再下载 Torch 或隐式调整依赖版本：

```bash
cd "$RHINOFORGE_SOURCE"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
python -m pip install '.[vision,vla,audio,dev]' --no-index --no-deps --no-build-isolation \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
python -m pip check
python -c 'from importlib.metadata import version; print("RhinoForge:", version("rhinoforge"))'
```

“一键部署”只覆盖依赖环境，仍需编译 RF 并准备独立资产。离线使用还需提前备齐
系统编译工具、SDK、Launch/算子、模型及输入。继续[模型准备](#model-setup)和运行；
后续在自己的 Git 分支中使用[同环境 editable 开发](#development)。
环境包的清单、第三方分发材料及已知限制见[依赖包说明](environment_bundle.zh.md)。

<a id="2-准备公开模型与本地配置"></a>
<a id="model-setup"></a>

## 3. 准备公开模型与本地配置

以下步骤由两条部署路径共用。保持所选环境已激活，并进入与该环境匹配的源码目录。

将固定 revision 的完整 checkpoint（config、tokenizer 和权重）下载到新缓存。
官方磁盘权重是 BF16，本例加载为 FP16，无需离线转换。`hf` 命令由项目依赖
`huggingface-hub` 提供。

```bash
cd "$RHINOFORGE_SOURCE"
export RPU_MODEL_CACHE="/absolute/path/to/new/model-cache"
export RHINOFORGE_RESULTS="/absolute/path/to/new/qwen3-results"
export RHINOFORGE_CONFIG="$RHINOFORGE_RESULTS/qwen3.local.toml"
mkdir -p "$RPU_MODEL_CACHE" "$RHINOFORGE_RESULTS"
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-0.6B"
python - <<'PY'
import json, os
from pathlib import Path
text = Path("examples/configs/qwen3/text/0_6b/fp16.toml").read_text()
text = text.replace("[run]\n", "[run]\noutput_dir = " + json.dumps(os.environ["RHINOFORGE_RESULTS"]) + "\n", 1)
text += '\n[logging]\nlevel = "INFO"\n'
Path(os.environ["RHINOFORGE_CONFIG"]).write_text(text)
PY
python -m examples.run --config "$RHINOFORGE_CONFIG" --check-config
```

`--check-config` 仅检查语法和 profile 关系，不加载权重、不解析动态库，也不检查
设备权限。模型许可和访问条款与 RhinoForge 源码许可分别适用。
也可使用同一固定 revision 的已有完整副本，但这不属于全新下载。

<a id="3-在空闲设备上运行"></a>
<a id="run-inference"></a>

## 4. 在空闲设备上运行

使用平台调度器/状态工具并与其他用户协调，确认板卡空闲。获取**其他板卡作业也
遵守的同一把锁**；自行创建不同路径的锁无法保护其他作业。若平台使用 `flock`，
填入管理员指定的锁路径（仅当平台确实采用此约定时，才使用 `/run/lock/rpu.device`）：

```bash
export RPU_DEVICE_LOCK="/absolute/path/to/platform/device.lock"
flock --nonblock "$RPU_DEVICE_LOCK" \
  "$RHINOFORGE_ENV/bin/python" -m examples.run --config "$RHINOFORGE_CONFIG"
```

获取锁失败就不要执行。使用调度器或专用启动器的平台应遵循相应机制。锁须覆盖
整个 Python 进程，包括 native 初始化与清理。推理账号须对一个 RPU 节点及
`/dev/mem` 有 `O_RDWR` 权限，见[权限检查](runtime_assets.zh.md#板卡-sdk-与设备权限)。
显式请求 RPU 的 CausalLM 加载会在读取完整权重前检查设备和 native backend；
CPU-only 加载不要求设备。

若管理员批准的执行策略要求 `sudo`，用显式环境保留**同一新 Python 环境、资产
和模型**。在源码根目录执行下列命令，替代上面的普通用户命令：

```bash
sudo env -i \
  HOME="$HOME" PATH="$RHINOFORGE_ENV/bin:/usr/bin:/bin" \
  PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  RHINO_LAUNCH_ROOT="$RHINO_LAUNCH_ROOT" \
  RPU_KERNEL_LIB_PATH="$RPU_KERNEL_LIB_PATH" \
  RPU_KERNEL_LIB_PATH_EXPANSION="$RPU_KERNEL_LIB_PATH_EXPANSION" \
  RPU_MODEL_CACHE="$RPU_MODEL_CACHE" \
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$RHINOFORGE_SOURCE" \
  /usr/bin/flock --nonblock "$RPU_DEVICE_LOCK" \
  "$RHINOFORGE_ENV/bin/python" -m examples.run --config "$RHINOFORGE_CONFIG"
```

先检查 `LD_LIBRARY_PATH` 保留当前 `$RHINOFORGE_ENV/lib`、所选 Launch 与必需 SDK 路径，
并去除旧环境路径。进程级 Git 配置用于
记录当前 checkout 的来源。特权执行可能生成 root 所有的私有结果子目录；进程结束
后，若平台允许，仅将本次新建任务结果归还调用用户：

```bash
sudo chown -R "$(id -u):$(id -g)" "$RHINOFORGE_RESULTS"
```

若要核实实际导入来源，应使用与推理相同的账号和环境；使用获准特权路径时，
此检查也应保留前述显式 `sudo env -i` 环境：

```bash
flock --nonblock "$RPU_DEVICE_LOCK" "$RHINOFORGE_ENV/bin/python" - <<'PYPROBE'
from pathlib import Path
import torch
import rpu_backend

print("Python package:", rpu_backend.__file__)
mapped = {
    line.split(maxsplit=5)[-1]
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "rpu_backend.so" in line
}
print("Loaded native libraries:", *sorted(mapped), sep="\n")
assert torch.rpu.is_available()
PYPROBE
```

Linux 进程映射显示实际加载的 native 库。此检查会导入 native runtime，须保持
板卡空闲并持有平台锁；`pip show` 不能单独证明加载的 native 库来源。
权限修复或更换资产后，应启动新 Python 进程。

<a id="4-解读短-case-结果"></a>
<a id="read-results"></a>

## 5. 解读短 case 结果

模板执行两次预热和三次正式调用（W2/R3）。固定 tokenizer 下裸提示词为 7 token
（P7）；D4 表示 prefill 首 token 后再执行四次 decode，因此**最多生成 5 token**，
遇 EOS 可更早结束。模板不应用 chat template，是短文本续写 smoke；重复部分提示词
本身不表示设备失败，也不构成完整问答。耗时不含模型加载和预热。

`INFO` 显示生成文本，默认 `ERROR` 隐藏普通输出。正常 Graph BUILD/REPLAY 是信息
日志；runner 会捕获并回放 native 输出，stderr 本身不等同于执行失败。
见[日志与结果说明](../examples/RESULTS.md)。

每次执行在 `RHINOFORGE_RESULTS` 下创建 `qwen3-0.6b.fp16.text-*` 子目录，包含
`report.json` 和最后一轮输出 `output.pt`；native 日志位于其父结果目录。
同时检查退出码和报告：`PASS` / `execution.status=COMPLETED` 表示执行完成；
普通 runner 仍记录 `correctness=not_run`、`formal_certification=false`，不会执行
CPU 参考对比或证明完整数值/任务认证。相关独立检查见[模型验证](validation_policy.zh.md)。

后续参考[模型资产](model_assets.zh.md)、[运行时配置](runtime_config.zh.md)和
[模型运行与调试](model_testing.zh.md)。

<a id="5-在同一环境中开发"></a>
<a id="development"></a>

## 6. 在同一环境中开发

继续使用相同的 `RHINOFORGE_ENV`、源码目录和配套运行时。需要 Python 修改直接
从源码生效时，将普通安装替换为可编辑安装；修改 native 代码后仍须重新执行构建。
主机编译工具与 `CC=/usr/bin/gcc`、`CXX=/usr/bin/g++` 沿用首次构建时的设置。
从 GitHub 指定 tag/commit 创建自己的开发分支；依赖包不携带另一份源码快照。
路径二已包含 `dev` 依赖：先重新执行第 2.2 节的兼容性检查，并在下方
`pip install` 命令中追加 `--no-index --no-deps`，确保开发安装不下载或改动依赖。
路径一按下方命令解析并安装尚缺的开发依赖。

```bash
cd "$RHINOFORGE_SOURCE"
python -m pip install -e '.[dev]' -c constraints/security.txt --no-build-isolation \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
python -m pip check
```

对应入口需要可选依赖时，使用 `'.[dev,vision]'`、`'.[dev,vla]'` 或 `'.[dev,audio]'`。
无需另建开发环境或更换 `RHINOFORGE_ENV` 变量。依赖已安装后，构建可交付 wheel：

```bash
export RHINOFORGE_WHEELS="$RHINOFORGE_WORKDIR/wheels"
python -m pip wheel . -c constraints/security.txt --no-build-isolation --no-deps \
  --wheel-dir "$RHINOFORGE_WHEELS" \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
```

记录源码提交和 diff、解析后的依赖、wheel/native hash 以及 Launch/算子/SDK
身份，并在新的部署目录验证 wheel。Python 和 native 修改后都要重新构建 wheel；
`.so` hash 不变不能证明旧 wheel 已包含当前 Python 代码。更新源码后重新安装/构建；
依赖和 ABI 未变化时可复用原环境，依赖变化时先核对新需求及环境关联记录。
制作公开依赖包时另建未安装 RhinoForge 的干净环境，不打包这里的普通或 editable 安装。

<a id="build-bundle"></a>

## 附录：可选音频工具与环境包维护

本节的音频工具用于在路径一环境中补齐音频场景；Qwen3 文本 FP16 不需要这些工具。
公开依赖包的制作设计见[独立说明](environment_bundle.zh.md)，它必须从全新、
未安装 RhinoForge 的依赖环境制作，不能打包下面已经用于开发/推理的环境。
依赖包不附带 Conda 管理器；这里的 `conda install` 命令供路径一用户使用。

需要全部三组扩展时，先安装 Conda 音视频库，再安装 Python extras。保持路径一的
packaging/setuptools 兼容范围，不让 pip 替换 Conda 管理的副本：

```bash
export RHINOFORGE_DOWNLOADS="${RHINOFORGE_DOWNLOADS:-$HOME/rhinoforge-downloads}"
mkdir -p "$RHINOFORGE_DOWNLOADS"
conda install --yes --override-channels --channel conda-forge \
  --prefix "$RHINOFORGE_ENV" ffmpeg libsndfile \
  "packaging>=24.2,<26" "setuptools>=71,<81"
cd "$RHINOFORGE_SOURCE"
python -m pip install '.[vision,vla,audio]' -c constraints/security.txt --no-build-isolation \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
ffmpeg -version
```

`conda-forge` 的 `linux-aarch64` 没有 `sox` 包。将
[官方 SoX 14.4.2 源码](https://sourceforge.net/projects/sox/files/sox/14.4.2/)
构建到**同一个** `RHINOFORGE_ENV`，不依赖系统已有的可执行程序。此构建将静态
`libsox` 链入 CLI，提供内建 PCM/WAV、音频效果与重采样，关闭外部 codec、插件和
播放设备，不包含 MP3 等被关闭的 SoX codec。其他格式使用随包的 FFmpeg/libsndfile
各自支持的能力；它们的存在不会自动为本 SoX 构建增加 codec。

下方子 shell 临时清除构建环境中的其他库路径，退出后仍保留当前激活的 Conda
环境。源压缩包和许可证保存在环境内部，供后续核对和分发审查。

```bash
export RHINOFORGE_SOX_BUILD="$RHINOFORGE_WORKDIR/sox-build"
mkdir "$RHINOFORGE_SOX_BUILD"
mkdir -p "$RHINOFORGE_DOWNLOADS"
curl --fail --location --retry 3 \
  https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz \
  --output "$RHINOFORGE_DOWNLOADS/sox-14.4.2.tar.gz"
printf '%s  %s\n' b45f598643ffbd8e363ff24d61166ccec4836fea6d3888881b8df53e3bb55f6c \
  "$RHINOFORGE_DOWNLOADS/sox-14.4.2.tar.gz" | sha256sum -c -
(
  export PATH=/usr/bin:/bin
  unset CPPFLAGS LDFLAGS LIBS CFLAGS LD_LIBRARY_PATH
  tar -xzf "$RHINOFORGE_DOWNLOADS/sox-14.4.2.tar.gz" -C "$RHINOFORGE_SOX_BUILD"
  cd "$RHINOFORGE_SOX_BUILD/sox-14.4.2"
  CC=/usr/bin/gcc CFLAGS=-O2 PKG_CONFIG=/usr/bin/false ./configure \
    --prefix="$RHINOFORGE_ENV" \
    --disable-shared --enable-static --disable-openmp \
    --without-libltdl --without-magic --without-png --without-ladspa \
    --without-mad --without-id3tag --without-lame --without-twolame \
    --without-oggvorbis --without-opus --without-flac \
    --without-amrwb --without-amrnb --without-wavpack \
    --without-sndio --without-coreaudio --without-alsa --without-ao \
    --without-pulseaudio --without-waveaudio --without-sndfile \
    --without-oss --without-sunaudio --without-mp3 \
    --without-gsm --without-lpc10 \
    --with-distro='RhinoForge portable PCM tooling'
  make -j2
  make install
  mkdir -p "$RHINOFORGE_ENV/share/licenses/sox" "$RHINOFORGE_ENV/share/sources/sox"
  cp COPYING LICENSE.GPL LICENSE.LGPL "$RHINOFORGE_ENV/share/licenses/sox/"
  cp "$RHINOFORGE_DOWNLOADS/sox-14.4.2.tar.gz" "$RHINOFORGE_ENV/share/sources/sox/"
  cat > "$RHINOFORGE_ENV/share/sources/sox/SOURCE.txt" <<'SOX_SOURCE'
SoX 14.4.2
Source: https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz
SHA256: b45f598643ffbd8e363ff24d61166ccec4836fea6d3888881b8df53e3bb55f6c
Static libsox linked into the CLI; optional external codecs, device backends,
plugins and OpenMP disabled. Built-in PCM/WAV formats and effects are included.
See share/licenses/sox/COPYING, LICENSE.GPL and LICENSE.LGPL.
SOX_SOURCE
)
"$RHINOFORGE_ENV/bin/sox" --version
"$RHINOFORGE_ENV/bin/sox" -n -r 16000 -c 1 -b 16 \
  "$RHINOFORGE_SOX_BUILD/tone-16k.wav" synth 0.05 sine 440
"$RHINOFORGE_ENV/bin/sox" "$RHINOFORGE_SOX_BUILD/tone-16k.wav" \
  -r 8000 "$RHINOFORGE_SOX_BUILD/tone-8k.wav"
"$RHINOFORGE_ENV/bin/sox" --i "$RHINOFORGE_SOX_BUILD/tone-8k.wav"
```

上述短命令先生成 16 kHz、16-bit 单声道 PCM 的 800 个采样，再重采样为 8 kHz 的
400 个采样。它只检查这条 CLI 路径，不代表所有音频模型或 codec 都已验证。

这些命令只补充音视频依赖，不会生成公开环境包。制作期间须保留 Conda 包缓存
直到打包完成；缓存不作为发布内容。保留包内所需的前缀迁移记录，使用独立的干净
依赖环境，并验证不同绝对路径下的部署、源码
构建和受影响的模型路径。第三方依赖应附带各自所需的许可证、声明和源码材料；
SoX 的材料不能代替 FFmpeg、libsndfile 等其他组件的材料。

旧完整包脚本继续只用于历史交付。新的 `pack_dependency_env.py` 只接收干净依赖
环境和审核后的第三方材料；制作命令和边界见[依赖包说明](environment_bundle.zh.md)。
