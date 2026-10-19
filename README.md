<div align="center">

<a href="https://github.com/HUIXI-AI/RhinoVLA"><img src="https://raw.githubusercontent.com/HUIXI-AI/RhinoVLA/3bfdbbfe8370be42f3a936eec616d065219e1825/assets/huixi_logo_cropped.png" alt="辉羲智能" height="72" /></a>

# RhinoForge

**面向 Rhino Processing Unit 的 PyTorch 推理与模型部署工具**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

[English](README_EN.md) | **简体中文**

</div>

RhinoForge 将 Rhino 设备接入为 `torch.rpu`。项目在同一个
PyTorch 后端中提供模型适配器、融合模型运行时、图捕获与重放、SPM 内存管理、
多核执行、离线量化和模型移植模板。

发行包名为 `rhinoforge`，Python 导入名仍为 `rpu_backend`。RhinoForge 仅支持
推理，不支持训练和微调。

当前源码版本为 **v1.3.0.dev0**（开发版），依赖已归档的 **Rhino Launch v1.2.0**；
使用[运行时资产指南](docs/runtime_assets.zh.md)指定的正式版组件。版本规则见[发布流程](docs/release_policy.md)。更新范围见[发行说明](CHANGELOG.md)，
已有环境请先阅读[升级指南](docs/upgrading_1_2.zh.md)。

## 主要能力

- **PyTorch 集成：** 通过标准自定义设备路径和 `torch.rpu` API 使用设备。
- **模型部署：** 通过模型适配器和 TOML 示例运行文本、视觉语言、视觉和 VLA
  配置。
- **可扩展运行时：** 移植新模型时可复用公开的融合模型、图、SPM、多核、批量
  提交和适配器机制。
- **按配置界定支持范围：** 每个状态都绑定到具体 checkpoint、精度、输入范围、
  配置和运行时资产组合。

## AI 辅助模型移植

仓库内置的 [`rhinoforge-port` skill](.agents/skills/rhinoforge-port/SKILL.md)
提供两种模式：**Assess** 只读分析能力差距，分别给出 outcome 与 certification；
**Port** 只从已接受、certified 且非 Blocked 的评估开始，并按模型类型加载公开
playbook。规范流程见[模型移植](docs/model_porting.zh.md)。

## 示例

[`examples/`](examples/README.md) 提供统一运行入口和按模型归档的 TOML 配置，覆盖
文本、图文、视觉组件和 VLA 的公开使用方式。目录结构、精度和输入要求见
[配置索引](examples/configs/README.md)。

调用方准备模型权重和请求数据；配置与运行时共同检查 checkpoint、精度、核数和输入范围。

Qwen3 ASR/TTS 使用独立语音入口，下载和运行步骤见[语音指南](docs/qwen3_audio.zh.md)。

## 快速开始

当前源码需要 Python 3.12（含开发头文件与 venv）、C++17 编译器、板卡 SDK/驱动，以及与源码匹配的
Rhino Launch 和算子资产。请按[当前源码构建说明](docs/development_build.md)
创建独立环境并安装依赖；版本以 [pyproject.toml](pyproject.toml) 为准。
视觉、VLA 和语音入口分别使用 `vision`、`vla`、`audio` 可选依赖。

1. 按构建说明选择实际可用的 `CC/CXX`，清除 `PYTHONPATH/RPU_BACKEND_SO`，
   校验并解开两层 Runtime 包，在独立环境中安装当前源码。
2. 按[模型资产说明](docs/model_assets.zh.md)准备模型 checkpoint，设置
   `RPU_MODEL_CACHE`，选择对应的 TOML 示例。
3. 按[入门指南](docs/getting_started.zh.md#3-在空闲设备上运行)确认板卡空闲，
   检查 `/dev/rpu*` 和 `/dev/mem` 的读写权限并使用平台设备锁，再运行推理。
   以下命令需在该锁和获准的执行账号下运行：

```bash
python -m pip check
cp examples/configs/qwen3/text/0_6b/fp16.toml qwen3.local.toml
cat >> qwen3.local.toml <<'TOML'

[logging]
level = "INFO"
TOML
python -m examples.run --config qwen3.local.toml --check-config
python -m examples.run --config qwen3.local.toml
```

示例显式设置 `logging.level="INFO"`，以显示生成文本；默认 `ERROR` 仅显示错误及独立摘要。
`--check-config` 只检查语法/profile，不检查设备、动态库或权重。
本例使用裸提示词 P7/D4、W2/R3，最多生成 5 token，不应用 chat template。
结果默认在 `perf_results/examples/qwen3-0.6b.fp16.text-*` 下；`PASS` 表示执行完成，
不代表已做 CPU 参考对比或完整质量认证。首次安装建议按入门指南把结果放在仓库外。
[模型运行与调试](docs/model_testing.zh.md)说明 TOML runner、量化与调试入口。

完整安装和首次运行步骤见[入门指南](docs/getting_started.zh.md)。

## 文档

| 主题 | 文档 |
|---|---|
| v1.2 更新与安装 | [发行说明](CHANGELOG.md) · [升级指南](docs/upgrading_1_2.zh.md) |
| 完整文档导航 | [文档索引](docs/README.md) |
| 当前源码安装 | [源码构建](docs/development_build.md) |
| 安装与首次运行 | [入门指南](docs/getting_started.zh.md) |
| 语音识别与生成 | [Qwen3 ASR/TTS](docs/qwen3_audio.zh.md) |
| 示例 | [示例目录与运行入口](examples/README.md) |
| 受限 Launch 与算子资产 | [受限运行时资产](docs/runtime_assets.zh.md) |
| 模型 checkpoint | [模型资产](docs/model_assets.zh.md) |
| Runtime/TOML 参数 | [运行时配置](docs/runtime_config.zh.md) |
| 模型运行、测试与调试 | [模型运行与调试](docs/model_testing.zh.md) |
| 适量验证 | [模型验证策略](docs/validation_policy.zh.md) |
| 运行时设计 | [架构](docs/architecture.zh.md) |
| Python 与 C++ API | [API 参考](docs/api_reference.zh.md) |
| 移植新模型 | [模型移植](docs/model_porting.zh.md) |
| 离线量化 | [量化](docs/quantization.zh.md) |
| 报告安全问题 | [安全策略](SECURITY.md) |
| 参与贡献 | [贡献指南](CONTRIBUTING.md) |

## 许可证

RhinoForge 源码采用 [Apache License 2.0](LICENSE)。模型 checkpoint、Rhino
Launch、主算子资产与扩展算子资产及其他第三方材料遵循各自条款，不属于本仓库 `LICENSE`
文件的授权范围。
