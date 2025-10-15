# NanoChat 知识沉淀

## 架构决策记录
- 采用分层目录（core/data/runtime/configs/interfaces/utils）以降低耦合并提升可维护性。
- 使用 `RuntimeMonitor` 抽象监控入口，避免在业务代码中硬编码具体监控实现。
- 保留 `nanochat/common.py` 作为兼容层，逐步引导至新的 `utils` 包。
- `nanochat.utils` 与 `nanochat.data` 采用懒加载入口，避免在未安装 `torch` 或 `pyarrow` 时导入失败。

## 常用命令
| 目标 | 命令 |
| --- | --- |
| 运行单元测试 | `pytest` |
| 运行代码规范检查 | `ruff --fix` / `mypy` |
| 训练基础模型 | `torchrun --nproc_per_node=8 -m scripts.base_train` |
| 启动 Web 服务 | `python -m scripts.chat_web -i sft` |

## 配置指南
- 使用 `nanochat/configs/loader.py` 读取 JSON/YAML/TOML 配置，并支持命令行覆盖。
- 未声明的配置键会抛出 `KeyError`，防止拼写错误或未知参数。
- `torchrun` 自动忽略 `--local_rank`、`--rank`、`--world_size` 等参数。
- `nanochat/configs/runtime.py` 提供数据类默认值，训练脚本示例（`BaseTrainConfig`）可直接生成可追踪的配置快照。

## 测试提示
- 可选依赖（`regex`、`rustbpe`、`tiktoken`）缺失时，`tests/test_rustbpe.py` 会自动跳过，保证配置解析等轻量测试仍能运行。
- 由于模块懒加载，未安装 `torch` 也能执行配置相关测试；在需要训练/推理时请确保正确安装 GPU/CPU 版本的 `torch`。
- 数据集工具新增依赖保护：缺少 `pyarrow` 或 `requests` 时会抛出明确异常，可通过 `pip install pyarrow requests` 安装。相关回归测试位于 `tests/test_dataset.py`。

## 故障排查
1. **推理超时**：`InferenceTimeout` 会在日志中记录详细上下文，可结合监控事件定位瓶颈。
2. **配置类型不匹配**：检查配置文件与默认值类型是否一致，必要时使用引号包裹字符串。
3. **缺少 tokenizer**：运行 `scripts.tok_train` 或从缓存目录复制既有 tokenizer。

## 下一步规划
- 拓展 `RuntimeMonitor`，支持导出 Prometheus/OpenTelemetry 指标。
- 为训练与评测脚本补充端到端集成测试。
- 编写发布流程与回滚预案，完善 Stage 4、Stage 5 目标。
