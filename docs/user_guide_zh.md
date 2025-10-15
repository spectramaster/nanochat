# NanoChat 中文使用手册

## 1. 环境准备
1. 安装 Python 3.10 及以上版本，推荐使用 `uv` 或 `pip` 创建虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安装项目依赖：
   ```bash
   pip install -e .
   pip install .[dev]  # 可选：安装开发与测试工具
   ```
3. 根据需要配置 CUDA 与 GPU 驱动，确保 `torch` 能够识别 GPU。

## 2. 目录结构速览
```
nanochat/
  core/        # 模型主体、损失函数、优化器
  data/        # 数据集、数据加载与分词工具
  runtime/     # 推理引擎、执行计划、分布式组件
  configs/     # 配置加载与兼容逻辑
  interfaces/  # Web UI、报告生成等接口
  utils/       # 日志、分布式、路径等通用工具
```
- 完整 API 请参阅 `nanochat/__init__.py`。
- 旧版 `common.py` 已转为兼容层，新功能请引用 `nanochat.utils`。

## 3. 常用脚本与配置
### 3.1 结构化配置
所有脚本支持通过 JSON/YAML/TOML 文件或命令行参数覆盖默认值：
```bash
python -m scripts.base_train --config=configs/train.yaml --device_batch_size=16 --run="demo"
```
- 若使用 `torchrun`，请将脚本参数放在 `--` 之后。
- 配置文件需为键值对结构，未在脚本默认值中出现的键会抛出错误。
- 训练脚本示例使用 `nanochat.configs.runtime.BaseTrainConfig` 提供数据类默认值，可与版本控制中的配置快照对齐。

### 3.2 训练流程
1. **预训练**：
   ```bash
   torchrun --nproc_per_node=8 -m scripts.base_train --run=pretrain-demo
   ```
2. **中期微调**：
   ```bash
   torchrun --nproc_per_node=8 -m scripts.mid_train --run=mid-demo --config=configs/mid.yaml
   ```
3. **监督微调 (SFT)**：
   ```bash
   torchrun --nproc_per_node=8 -m scripts.chat_sft --run=sft-demo
   ```
4. **强化学习 (RL)**：
   ```bash
   torchrun --nproc_per_node=8 -m scripts.chat_rl --run=rl-demo
   ```

### 3.3 推理与评测
- 命令行对话：`python -m scripts.chat_cli -i sft`
- Web 服务：`python -m scripts.chat_web -i sft --port=8000`
- 评测基准：`torchrun --nproc_per_node=8 -m scripts.base_eval`

## 4. 数据与分词
- 数据集下载：使用 `nanochat.data.dataset.download_dataset()` 或运行 `python -m nanochat.data.dataset`。
- 依赖说明：数据下载依赖 `requests`，读取 parquet 依赖 `pyarrow`；请根据需要 `pip install requests pyarrow`。
- Tokenizer 训练：`python -m scripts.tok_train --vocab_size=65536`
- 预置 tokenizer 与 token bytes 默认存放于 `~/.cache/nanochat/tokenizer/`。

## 5. 日志与监控
- 项目默认使用彩色日志格式，可通过环境变量 `NANOCHAT_BASE_DIR` 调整缓存目录。
- 分布式训练日志仅在 rank=0 输出，辅助函数 `print0` 位于 `nanochat.utils.display`。
- 如需更丰富的监控，可将 `nanochat.utils.logging.setup_default_logging` 替换为自定义实现。

## 6. 测试与质量保障
1. 单元测试：`pytest`
2. 静态检查：`ruff --fix`、`mypy`
3. 一键执行（建议使用 `pre-commit`）：
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```
4. 未安装 `torch` 时，依托懒加载机制仍可运行配置相关测试；缺少 `regex` / `rustbpe` / `tiktoken` 时，相关 tokenizer 测试会自动跳过。

## 7. 常见问题
1. **无法加载配置文件**：确认文件后缀为 `.json` / `.yaml` / `.toml`，并符合键值结构。
2. **`torchrun` 报未知参数**：使用 `--` 分隔 `torchrun` 参数与脚本参数，例如 `torchrun ... -- -i sft`。
3. **下载数据超时**：`nanochat.data.dataset.download_single_file` 内置指数回退，可重复运行，确保网络畅通。
4. **Tokenizer 缺失**：运行 `scripts.tok_train` 后重新执行训练脚本。

## 8. 进一步阅读
- 架构设计：`docs/standards/architecture_principles.md`
- 优化路线图：`docs/optimization_plan.md`
- 测试用例：`tests/` 目录

如有问题，欢迎在 issue 或 PR 中反馈，共同完善 NanoChat。
