# 交接文档：Token in Token out

> 关键词：`/inference/v1/generate`、token_ids 进 / token_ids 出、`skip_tokenizer_init`、`return_token_ids`、`detokenize=False`、RL rollouts、Disaggregated 协调器。

## 1. 这是什么 / 为什么需要

“Token in token out”指**直接以 token_ids 作为输入、直接返回生成的 token_ids**的推理路径，跳过服务端的 tokenize / detokenize。

为什么需要：

- **RL/RLHF**：trainer 通常已经把 prompt tokenize 好了，希望拿回原始 token 序列，省去字符串往返；
- **避免 retokenize 漂移**：推理侧 detokenize 成文本、训练侧再 tokenize 回去，可能得到**不一致**的 token_ids；
- **性能**：外部已有 tokenizer 时，省去 tokenizer 初始化与 detokenize 开销。

> 重要定位：**该特性的主体能力在上游 vLLM**，vLLM Ascend 作为硬件插件**原样继承**，并在 NPU 代码路径上做适配与验证；
> Ascend 侧没有为它新增 `VLLM_ASCEND_*` 环境变量。

实际涉及三种相关但不同的机制，交接时建议分清楚：

| 机制 | 行为 | 主要开关 |
| --- | --- | --- |
| **严格 token-in-token-out** | 不初始化 tokenizer/detokenizer，输入必须是 token_ids，输出是 token_ids | `--skip-tokenizer-init` / `skip_tokenizer_init=True` |
| **专用 generate 端点** | `/inference/v1/generate` 收 `token_ids`、回 `token_ids`（Disaggregated-Everything 协调器契约） | 该端点本身（上游 serve 入口） |
| **返回 token_ids（仍带文本）** | 正常 tokenize，但 API 额外返回 `prompt_token_ids` + `token_ids` | 请求体 `return_token_ids=True` |
| **逐请求跳过 detokenize** | tokenizer 存在，但单个请求不转文本 | `SamplingParams(detokenize=False)` |

## 2. `/inference/v1/generate` 端点（Ascend 重点验证对象）

这是团队在 Ascend 上重点做 E2E 验证的 token-in-token-out 入口，对标上游
`tests/entrypoints/serve/disagg/test_serving_tokens.py`，用于 Disaggregated-Everything 协调器。

请求/响应契约（来自 Ascend E2E 测试）：

```bash
POST /inference/v1/generate
{
  "model": "Qwen/Qwen3-0.6B",
  "token_ids": [1, 2, 3],
  "sampling_params": {"max_tokens": 5, "temperature": 0.0},
  "stream": false
}
# 响应：choices[0].token_ids 为生成的 token_ids
```

E2E 校验的四个不变式（见 `test_generate_tokens.py`）：

1. 传 `token_ids` 能拿回生成 `token_ids`，并遵守 `max_tokens`；
2. 不传 `max_tokens` 时，**不能**静默截断在 dataclass 默认值 16，服务端会用 `max_model_len` 补默认值；
3. 流式（stream）与非流式在贪心（greedy）下产出的 token 序列**完全一致**；
4. 对同一 token prompt，`/inference/v1/generate`（`detokenize=False`）解码出的文本与 `/v1/chat/completions` 的输出**一致**，证明 token 路径与标准 OpenAI 兼容路径自洽。

> 现状提示：该端点的 **Ascend E2E 测试目前在 `dp_in_dp_out` 分支**
> （`tests/e2e/pull_request/light/one_card/test_generate_tokens.py`），尚未合入 `main`。
> 测试用例特意关闭 prefix caching（`--no-enable-prefix-caching`），因为 cache 命中/未命中会走不同 GEMM 形状、翻转 argmax，破坏上面 3、4 的一致性断言。

## 3. 上游开关与配置

### A. 服务/引擎级（严格 token-in-token-out）

```bash
vllm serve <model> --skip-tokenizer-init
```

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen3-0.6B", skip_tokenizer_init=True)
outputs = llm.generate([{"prompt_token_ids": [1, 2, 3, 4]}])
# outputs[0].outputs[0].token_ids  → 生成的 token_ids
```

输入可用 `TokensPrompt(prompt_token_ids=[...])`、`{"prompt_token_ids": [...]}`，
或 OpenAI `/v1/completions` 把 `prompt` 传成 int 列表（需 `--skip-tokenizer-init`）。

### B. API 级（仍用 tokenizer，但返回 token_ids）

```json
POST /v1/completions
{"model": "...", "prompt": "Hello world", "max_tokens": 20, "return_token_ids": true}
```

响应额外含 `prompt_token_ids` 与 `token_ids`；chat completions 同理。
相关项：`return_tokens_as_token_ids=true` 让 logprob 的 token 以 `"token_id:12345"` 字符串呈现（无 tokenizer 或 token 不可 JSON 编码时需要）。

### C. 逐请求跳过 detokenize（离线）

```python
SamplingParams(max_tokens=10, temperature=0.0, detokenize=False)
# 输出 text == ""，token_ids 与 detokenize=True 时一致
```

### D. Ascend 侧配置

**无 Ascend 专用环境变量**。所有配置都从上游 vLLM 继承。当前固定的上游版本见 `.github/vllm-release-tag.commit`。

## 4. 架构与调用链

```text
Client / RL Trainer
  ├─ token_ids 进 ──► serve 入口
  │     ├─ /inference/v1/generate（Disaggregated 协调器契约：收 token_ids、回 token_ids）
  │     ├─ skip_tokenizer_init=True ──► Processor 只接受 prompt_token_ids
  │     └─ return_token_ids=True   ──► OpenAI serving 把 token_ids 附到响应
  └─► Scheduler / EngineCore ─► NPUModelRunner(vllm_ascend) ─► AscendSampler
        └─ Request.output_token_ids 更新 ─► RequestOutput.token_ids
             ├─ detokenize=True / 有 tokenizer ─► Detokenizer → text
             └─ detokenize=False / skip_tokenizer_init ─► 只回 token_ids
```

vLLM Ascend **不**对 token-in-token-out 做特例化，而是在整条 v1 引擎路径上**天然以 token_ids 运作**。关键落点（行号为参考）：

- **Scheduler**：token 计数基于 `prompt_token_ids + output_token_ids`（`vllm_ascend/core/recompute_scheduler.py`、`patch_balance_schedule.py`）。
- **Input batch**：跟踪每请求 output token 历史用于 penalties（`vllm_ascend/worker/npu_input_batch.py` 约 213-237 行）。
- **Model runner**：采样后扩展 `req_state.output_token_ids`（`vllm_ascend/worker/model_runner_v1.py` 约 2748-2755 行）；采样前 `update_async_output_token_ids()` 同步（约 2609-2612 行），异步调度 + penalties 时关键。
- **Sampler**：penalties 用 `prompt_token_ids` 与 `output_token_ids`（`vllm_ascend/sample/sampler.py` 约 60-69 行），并断言 `prompt_token_ids is not None`。
- **API 统计 patch**：`vllm_ascend/patch/platform/patch_chat_usage_accounting.py`（约 153-157 行）在 `return_token_ids` 时记录 `raw_output_token_ids`，用于 reasoning token 用量统计——**仅统计，不启用该特性**。

## 5. 入口与上游集成

- vLLM Ascend 作为平台插件注册（`setup.py` 的 `vllm.platform_plugins`），**不**覆写 tokenizer/detokenizer 入口；token-in-token-out 行为全部来自上游。
- **CI（在 NPU 上跑上游测试）**：`E2E-upstream` workflow 拉取上游 vLLM 固定 tag，注入 Ascend patch 后运行上游测试，`upstream_config.yaml` 中相关：
    - `tests/entrypoints/openai/completion/test_token_in_token_out.py`（严格模式 + `--skip-tokenizer-init`）
    - `tests/entrypoints/openai/test_return_token_ids.py`（`return_token_ids`）
    - `tests/detokenizer/test_disable_detokenization.py`（`detokenize=False`）
- **本仓库 E2E helper**：`tests/e2e/conftest.py` 支持把 token_ids 当 prompt 传入（list → `prompt_token_ids`），并在输出收尾时同时取 `token_ids` 与 `text`。

## 6. 测试

| 测试位置 | 覆盖 | 是否严格 token-in-token-out |
| --- | --- | --- |
| `tests/e2e/pull_request/light/one_card/test_generate_tokens.py`（**`dp_in_dp_out` 分支**） | `/inference/v1/generate` token 进 token 出、max_tokens 默认、流式一致、与 chat 一致 | 是（端点级） |
| `.github/workflows/scripts/upstream_config.yaml` | 列出上游 token-in-token-out 相关测试在 NPU 上跑 | 引用上游测试 |
| `tests/ut/worker/a2/test_model_runner_v1_with_device.py` 等 | `skip_tokenizer_init=True` | 仅测试便利（避免加载真实 tokenizer） |

`main` 上**没有**独立命名为 `test_token_in_token_out.py` 的 Ascend 测试文件；端点级 E2E 在 `dp_in_dp_out` 分支。

## 7. 环境变量

`vllm_ascend/envs.py` 中**无**与 token-in-token-out / skip-tokenizer / return_token_ids 相关的变量。相关但属通用 RL 的：`VLLM_BATCH_INVARIANT=1`（确定性 rollouts）、`VLLM_SERVER_DEV_MODE=1`（RL 示例的权重传输端点）。

## 8. 已知约束、坑与 TODO

### 上游约束（Ascend 继承）

1. **结构化输出与 `skip_tokenizer_init` 不兼容**（无 tokenizer 时会抛 `ValueError`）。
2. **工具调用 / reasoning 解析需要 tokenizer**：`skip_tokenizer_init=True` 时相关路径会报错。
3. **无 tokenizer 取 logprobs**：必须 `return_tokens_as_token_ids=True`，否则报 “Unable to get tokenizer …”。
4. **chat 模板**：`skip_tokenizer_init` 下模板处理需**在外部**完成，传入已 tokenized 的 `prompt_token_ids`。
5. **多模态**：token-in-token-out + 多模态在 Ascend 文档/示例中未覆盖；`pcp_utils.py` 假设存在 `prompt_token_ids` 用于 mRoPE 重算。

### Ascend 侧注意

1. `docs/` 中**暂无**对外的 token-in-token-out 专门文档。
2. 现有 RL 示例（`examples/rl/rlhf_http_*.py`）走的是**文本** prompt，不是 token-in-token-out。
3. penalties 开启时，Ascend sampler 断言 `sampling_metadata.prompt_token_ids is not None`，由 scheduler 负责填充（上游职责）。
4. 异步调度 + penalties：Ascend 显式在采样前 `update_async_output_token_ids()` 同步 output token，RL rollouts 用到 repetition/frequency penalty 时需关注。
5. `/inference/v1/generate` 的 Ascend E2E 目前在 `dp_in_dp_out` 分支，合入 `main` 前请同步迁移测试与 `test_config.yaml` 注册项。

## 9. 文件索引与相关提交

| 路径 | 关注点 |
| --- | --- |
| `tests/e2e/pull_request/light/one_card/test_generate_tokens.py`（`dp_in_dp_out`） | `/inference/v1/generate` E2E |
| `tests/e2e/conftest.py` | token_ids 作为 prompt（约 995-1003）、输出收尾取 token_ids+text（约 967-971） |
| `vllm_ascend/worker/model_runner_v1.py` | 2609-2612、2748-2755 |
| `vllm_ascend/worker/npu_input_batch.py` | 213-237 |
| `vllm_ascend/sample/sampler.py` | 60-69 |
| `vllm_ascend/core/recompute_scheduler.py` | 171-176 |
| `vllm_ascend/patch/platform/patch_chat_usage_accounting.py` | 153-157 |
| `.github/workflows/scripts/upstream_config.yaml` | 上游 token-in-token-out 测试登记 |

相关提交（`dp_in_dp_out` 分支）：

- `45b42a49` [Test] Add E2E test for /inference/v1/generate token-in-token-out on NPU
- `88a78d13` test: add e2e cases for token-in-token-out and X-data-parallel-rank
- `a325ebac` test: drop duplicate token-in-token-out e2e case
