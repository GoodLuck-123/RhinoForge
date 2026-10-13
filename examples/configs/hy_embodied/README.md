# Hy-Embodied UMI 配置

[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

本目录通过 `HyEmbodiedPolicy` 运行三相机 UMI FP16 配方：输入图片、指令和状态，输出 50 步、20 维归一化动作。两相机、不同文本长度和 W8/NVFP4 配方见 [HyVLA 配置](../hy_vla/README.md)。

## 配置说明

<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

| 配置 | 计算范围 | 权重获取 |
|---|---|---|
| [umi/fp16.toml](umi/fp16.toml) | 三相机、P240、UMI FP16，十步去噪 | [腾讯官方 BF16 权重](#weights)，与 HyVLA 配方共用父资产 |

<a id="fp16-基准做了什么"></a>

### 精度与执行

Vision、Language 和 Action 的主投影使用 FP16 权重与激活，patch/merger/norm/bias 为浮点；CPU embedding、状态和时间输入边界采用 FP32。模型内部动作网格为 `[50,32]`，公共入口返回有效的前 20 维 `[50,20]`。

模板的 `[example.opt_in]` 为空，运行时仍会应用以下默认优化：

| 阶段 | 已开启的执行方式 |
|---|---|
| Vision | 多视角打包；多核 patch embedding 与融合 merger 放入 Graph；Q 就地计算，K/V 按 16 行对齐 |
| Language | 保留模型与 Graph，使用 fast replay；partial RoPE 复用准备好的位置数据 |
| Action | 十步 Euler 去噪展开在图内；Attention 使用 TP8；缓存 mask、融合 SiLU×Mul；partial RoPE 和 fast replay |

这些设置由 [HyVLA 运行时默认项](../../../python/rpu_backend/adapters/hy_vla/runtime.py)提供。加载接口见 [HyEmbodiedPolicy](../../../python/rpu_backend/api/hy_embodied.py)，模板调用见 [runner](../../runners/hy_vla/runner.py)。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="quantization"></a>
<a id="权重准备"></a>

### 获取权重

`example.registry_alias="hy-embodied-0.5-vla-umi"` 读取 `$RPU_MODEL_CACHE/Hy-Embodied-0.5-VLA-UMI`。目录保留模型和 tokenizer；归一化文件用于需要物理动作解码的调用方，本模板输出归一化动作。

源权重为 [tencent/Hy-Embodied-0.5-VLA-UMI](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI/tree/3f53d1f8d2bc587c523cfdc9f1041ceee42c2524)，固定 revision 为 `3f53d1f8d2bc587c523cfdc9f1041ceee42c2524`。在仓库根目录下载：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download tencent/Hy-Embodied-0.5-VLA-UMI \
  --revision 3f53d1f8d2bc587c523cfdc9f1041ceee42c2524 \
  --local-dir "$RPU_MODEL_CACHE/Hy-Embodied-0.5-VLA-UMI"
```

FP16 模板直接消费这个目录，加载时转换，不产生新的权重文件。

<a id="使用"></a>
<a id="run-fp16"></a>

### 输入与运行

在已安装项目依赖的设备环境中，按 head、left wrist、right wrist 的顺序提供三张图片，填写配对指令和状态：

```bash
cp examples/configs/hy_embodied/umi/fp16.toml /tmp/hy_umi_fp16.toml
# 修改 input.images、input.instruction、input.state。
# 在 [run] 中添加 output_dir，保留 prefix_len=240。
vi /tmp/hy_umi_fp16.toml
python examples/run.py --config /tmp/hy_umi_fp16.toml --check-config
python examples/run.py --config /tmp/hy_umi_fp16.toml
```

<a id="输入与执行参数"></a>

### 常用参数

| 字段 | 模板值 | 省略时 | 范围与关联 |
|---|---|---|---|
| `input.images` | 三张图片占位路径 | 必填 | 按 head / left wrist / right wrist 排序 |
| `input.instruction` | `Move the object to the target.` | 必填 | 完整指令，由模型 tokenizer 处理 |
| `input.state` | 32 维零向量 | Policy 使用零状态 | 已归一化状态，最多 32 维，右侧补零；与图片和指令配对 |
| `input.noise_seed` | 0 | 0 | 初始去噪噪声的种子 |
| `input.prefix_len` | 240 | 240 | 前缀容量；16 的倍数，范围 16–240，并覆盖真实前缀 |
| `cpu.threads` | 未配置 | Policy 构建使用 12 | 此 runner 未向 Policy 传入线程数；构建时使用 API 默认 12，覆盖公共入口加载前设置 |

模板的 `run.warmup/runs/seed` 为 2 / 3 / 0。通用字段的默认值、输出目录、计时与诊断选项见[公共参数说明](../PARAMETERS.md)。

十步去噪固定在图中。切换相机数或精度时，选择 [HyVLA 对应模板](../hy_vla/README.md)。

## 输出

`run.output_dir` 保存 `report.json` 与 `[50,20]` 归一化动作快照。机器人坐标变换和动作执行由下游处理。
