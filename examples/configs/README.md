# 示例配置

[完整配置索引](CONFIGURATIONS.md) · [权重与转换](WEIGHTS.md) · [登记支持矩阵](../../docs/model_support.zh.md) · [所有入口](../README.zh.md)

从模型指南选择模板，查看配置差别和输入限制，再填写资产路径运行。

## 按模型选择

| 模型族 | 配置说明 | 主要区别 |
|---|---|---|
| G0.5 | [g05](g05/README.md) | 2/3cam × FP16、W8、W8+Action NVFP4；ACC/text 在文件内修改 |
| Pi0.5 | [pi05](pi05/README.md) | 2/3cam × 四精度；TP8 优化与 TP4/6 基线分组 |
| HyVLA | [hy_vla](hy_vla/README.md) | 2/3cam × 三精度；当前优化入口为 TP8 |
| Wall-OSS | [wall_oss](wall_oss/README.md) | base 按相机/分辨率/精度；RTC 与分拣为不同负载 |
| GR00T | [gr00t](gr00t/README.md) | 224/256 × FP16/W8，text 在文件内修改 |
| RhinoVLA | [rhinovla](rhinovla/README.md) | v1、v3 factory、独立验证片段分别使用 |
| Qwen3 | [qwen3](qwen3/README.md) | 模型大小、精度、Instruct / lm_head recipe |
| Qwen3.5 | [qwen3_5](qwen3_5/README.md) | text/VL、Dense/MoE、精度、固定 P/D 复现 |
| Qwen3-VL | [qwen3_vl](qwen3_vl/README.md) | text/VL、模型大小、精度、固定 P/D 复现 |
| DINOv3 | [dinov3](dinov3/README.md) | ViT-B FP16 图像编码 |
| Gemma4 | [gemma4](gemma4/README.md) | E4B FP16 |
| Llama | [llama](llama/README.md) | 3.2-1B FP16 |
| HY Embodied | [hy_embodied](hy_embodied/README.md) | 原 UMI fixture；优化模板见 HyVLA |
| HALO | [halo](halo/README.md) | Action expert；延期状态保留 |
| InternVLA | [internvla](internvla/README.md) | NavDP fixture |
| LingBot | [lingbot](lingbot/README.md) | v1/v2 分开；v1 延期状态保留 |

## 目录与模板规则

统一使用 `<模型族>/<版本、组件或模型大小>/<确实不同的输入几何>/<精度>.toml`；没有这些维度时省略对应层。
模型身份、权重 recipe、相机/尺寸等实质差异决定模板路径。ACC、text len、CPU、warmup/runs、AUTO chunk 等支持的参数直接修改文件，不复制每种参数组合。

| 分组 | 用途 |
|---|---|
| 普通模型目录 | 可选运行模板；推荐配置见模型指南 |
| 固定 P/D 模板 | 输入生成、profile 或计时方式不同的固定负载运行；保留原路径 |
| `legacy/` 或 legacy 文件 | 历史输入与接口兼容，具体限制见同族说明 |
| `controlled/` | 受控验证；Rhino v3 的 `[env]` 片段使用独立验证入口 |
| `unoptimized/` | 未启用对应优化的执行配置 |

## 参数与运行

统一运行命令为 `python examples/run.py --config <TOML>`，`--config` 必填。CPU 线程使用 `[cpu].threads`；省略时使用对应 runner 的默认值。ACC、文本长度和采样次数的合法值见模型指南，在同一文件修改。

```bash
python examples/run.py --config examples/configs/g05/3cam/fp16.toml --check-config
python examples/run.py --config examples/configs/g05/3cam/fp16.toml
```

按模型 README 替换资产、输入和库路径。`--check-config` 无需 Torch/设备，只检查语法及身份；不加载权重或访问设备。
REF 依赖使用 [当前两份正式库](../../docs/runtime_assets.md)，配置不依赖实验 REF。

<a id="本轮路径迁移"></a>
Rhino v3 新命令使用 [expert_w8a16.toml](rhinovla/v3/expert_w8a16.toml)，旧 `w8a16.toml` 保留为兼容别名。完整路径映射与输入迁移见 [v1.2.0 升级指南](../../docs/upgrading_1_2.zh.md)。

`[logging]` 与 `[run]` 同级，默认 `level="ERROR"`、`summary=true`。等级和输出文件见[日志说明](../README.zh.md#logging-and-performance-output)。
