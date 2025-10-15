# NanoChat 基线评估报告

## 1. 代码拓扑分析
- **核心模型层**：`nanochat/core/gpt.py`（Transformer 模型定义）与 `nanochat/core/loss_eval.py`、`nanochat/core/adamw.py` 紧密耦合，全部通过 `nanochat/runtime/engine.py` 暴露给训练与推理脚本。
- **数据层**：`nanochat/dataset.py`、`nanochat/dataloader.py` 与 `nanochat/tokenizer.py` 之间存在交叉导入，部分函数直接访问全局常量，缺乏明确的封装边界。
- **运行时层**：`nanochat/runtime/engine.py`、`nanochat/runtime/execution.py` 与 `nanochat/runtime/muon.py` 互相依赖共享状态，同时依赖 `nanochat.utils` 中的分布式初始化逻辑。
- **通用工具层**：`nanochat/utils` 集成了日志、分布式、路径与 Banner 打印等通用职责，避免单文件过载。
- **接口层**：`nanochat/report.py`、`nanochat/ui.html`、`nanochat/logo.svg` 等文件混杂在包根目录，没有统一归类。
- **配置层**：`nanochat/configurator.py` 采用动态执行脚本的方式加载配置，缺乏类型校验与默认值管理。

## 2. 功能矩阵
| 功能域 | 主要模块 | 痛点 |
| --- | --- | --- |
| 模型训练 | `engine.py`、`gpt.py`、`dataset.py` | 配置难以复现，调试需遍历多个文件 |
| 推理服务 | `engine.py`、`report.py` | 缺乏标准接口定义，推理请求结构不清晰 |
| 数据处理 | `dataset.py`、`tokenizer.py` | 工具函数散落，难以在其他项目复用 |
| 配置管理 | `configurator.py` | 动态执行不可控，无法校验参数类型 |
| 日志监控 | `common.py`、脚本级日志 | 日志格式不统一，缺少上下文信息 |
| 文档支持 | `README.md`、`docs/optimization_plan.md` | 缺少架构说明、API 文档与使用手册 |

## 3. 配置现状
- `configurator.py` 依赖 `exec` 与全局变量覆写策略，缺乏结构化配置；
- 多处脚本手动解析命令行参数，未提供统一入口；
- 配置与代码逻辑紧耦合，难以进行静态分析或工具支持。

## 4. 初步结论
- 需要通过目录分层与模块职责拆分降低耦合；
- 引入结构化配置与类型校验，保障可维护性；
- 建立统一日志、测试与文档体系支撑工程化落地。
