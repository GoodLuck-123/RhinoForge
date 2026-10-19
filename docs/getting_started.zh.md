# 入门指南

简体中文 | [English](getting_started.md)

本指南在新 Python 环境中安装当前 RhinoForge 源码，并运行 Qwen3-0.6B FP16。
板卡 OS、驱动和 SDK 是明确的外部平台前提，不由 Python 包或 Runtime 包安装。

## 1. 准备运行时和独立环境

1. 将源码克隆到新目录，记录 `git rev-parse HEAD`。
2. 按[Python 与编译器准备](development_build.md#prepare-python-and-the-compiler)
   安装 Python 3.12、开发头文件和 venv，并选择 `RHINOFORGE_PYTHON`、`CC` 和 `CXX`。
   该指南也提供系统软件仓库没有 Python 3.12 时的源码安装方法。
3. 按[运行时校验及双层解包](runtime_assets.zh.md#配置路径)设置
   `RHINO_LAUNCH_ROOT`、`RPU_KERNEL_LIB_PATH`、`RPU_KERNEL_LIB_PATH_EXPANSION`
   和动态库路径，确认提供方的 SDK 已安装并检查[设备权限](runtime_assets.zh.md#板卡-sdk-与设备权限)。

以下填入新的绝对路径，在同一 shell 继续。不要使用 `--system-site-packages`；
即使创建了新 venv，也应清除继承的覆盖变量：

```bash
export RHINOFORGE_SOURCE="/absolute/path/to/RhinoForge"
export RHINOFORGE_ENV="/absolute/path/to/new/rhinoforge-env"
export RHINOFORGE_BUILD="/absolute/path/to/new/rhinoforge-build"
unset PYTHONPATH PYTHONHOME RPU_BACKEND_SO
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
"$RHINOFORGE_PYTHON" -m venv "$RHINOFORGE_ENV"
. "$RHINOFORGE_ENV/bin/activate"
cd "$RHINOFORGE_SOURCE"

python -m pip install -c constraints/security.txt "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install . -c constraints/security.txt --no-build-isolation \
  -Cbuild-dir="$RHINOFORGE_BUILD" \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
python - <<'PY'
import sys
from importlib.metadata import version
from importlib.util import find_spec
assert sys.prefix != sys.base_prefix
print("Python:", sys.executable)
print("RhinoForge:", version("rhinoforge"))
print("Package source:", find_spec("rpu_backend").origin)
PY
```

末尾检查只读包元数据，不加载 native extension。依赖以 [pyproject.toml](../pyproject.toml)
为准；视觉模型将安装命令中的 `.` 改为 `'.[vision]'`，策略模型改为 `'.[vla]'`，
[语音模型](qwen3_audio.zh.md)改为 `'.[audio]'`；Qwen3 文本 FP16 无需这些可选依赖。
首次安装保留依赖解析。可编辑安装和 wheel 见[源码构建指南](development_build.md)。
板卡无法联网时可从获准的联网机器下载、校验后传输，Python 包须匹配板卡架构和
Python 3.12，并记录使用的镜像源。

## 2. 准备公开模型与本地配置

将固定 revision 的完整 checkpoint（config、tokenizer 和权重）下载到新缓存。
官方磁盘权重是 BF16，本例加载为 FP16，无需离线转换。`hf` 命令由项目依赖
`huggingface-hub` 提供。

```bash
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

## 3. 在空闲设备上运行

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

先检查 `LD_LIBRARY_PATH` 只保留所选 Launch 与必需 SDK 路径。进程级 Git 配置用于
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

## 4. 解读短 case 结果

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
