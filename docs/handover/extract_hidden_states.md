# 交接文档：Extract Hidden States

> 关键词：投机解码（spec decode）复用、hidden states 抽取、EAGLE 训练数据采集、KV connector 落盘。

## 1. 这是什么 / 为什么需要

`extract_hidden_states` 是一种**特殊的投机解码模式，但它并不真正做投机**。它复用 spec decode 的整套管线，达成以下目的：

1. 跑一遍**目标模型（target model）**，收集指定若干 transformer 层的 hidden states；
2. 把这些 hidden states 喂给一个极轻量的、**只写 cache 不算 attention** 的“草稿模型” `ExtractHiddenStatesModel`，将其写入一块专用的 KV cache；
3. 在请求结束时，由 `ExampleHiddenStatesConnector` 读取这块 cache，并把结果写成 `.safetensors` 文件落盘。

**核心用途**：离线采集 **EAGLE 类草稿模型的训练数据**（即目标模型在指定层的辅助 hidden states）。

每个请求产出的产物（safetensors）包含：

| 字段 | 形状 / 内容 |
| --- | --- |
| `hidden_states` | `(num_tokens, num_layers, hidden_size)` |
| `token_ids` | prompt 的 token_ids（用于回环校验） |
| 落盘路径 | `output.kv_transfer_params["hidden_states_path"]` |

> 注意：该模式每个请求只产出 1 个 output token，真正有价值的产物是落盘的 hidden states，而非生成文本。

主体逻辑在**上游 vLLM**（proposer / 草稿模型 / connector）；vLLM Ascend 侧主要贡献的是一个 **NPU 适配版 proposer 子类**，外加 v1 model runner 里的若干集成点。本特性**没有**新增任何 `VLLM_ASCEND_*` 环境变量。

## 2. 如何开启与配置

通过 `LLM(...)` 的 `speculative_config` 与 `kv_transfer_config` 两个参数开启。

### `speculative_config`

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `method` | `"extract_hidden_states"` | 自动解析 `model`，无需单独的草稿模型仓库 |
| `num_speculative_tokens` | **固定为 `1`** | 不做真正多 token 投机，上游会断言为 1 |
| `draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids` | 例如 `[2, 18, 34]` | 要从目标模型抽取的层索引 |

### `kv_transfer_config`

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `kv_connector` | `"ExampleHiddenStatesConnector"` | 落盘必需 |
| `kv_role` | `"kv_producer"` | 只生产、不消费 |
| `kv_connector_extra_config.shared_storage_path` | 目录路径 | 每个请求落一个 `.safetensors` |

### 最小可用示例

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    enable_chunked_prefill=False,
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [2, 18, 34],
            }
        },
    },
    kv_transfer_config={
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": "/path/to/output"},
    },
)

outputs = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=1))
path = outputs[0].kv_transfer_params["hidden_states_path"]
# safetensors 内含 key: hidden_states, token_ids
```

对外使用文档参见 `docs/source/user_guide/feature_guide/speculative_decoding.md` 的「Extracting Hidden States」一节。

## 3. 架构与调用链

```text
LLM.generate()
  └─ NPUModelRunner (v1)
       ├─ get_spec_decode_method("extract_hidden_states") → AscendExtractHiddenStatesProposer
       ├─ target.set_aux_hidden_state_layers(layer_ids)        # 配置抽取层
       ├─ target.forward() → (hidden_states, aux_hidden_states[])
       ├─ proposer.propose(sampled_ids, aux_hidden_states)
       │     └─ ExtractHiddenStatesModel.forward(stacked hidden states) → 写入 KV cache
       └─ finalize_kv_connector()
              └─ ExampleHiddenStatesConnector：请求结束时从 cache 取出 → 异步落盘 .safetensors
```

### 关键类与文件

| 组件 | 位置 | 角色 |
| --- | --- | --- |
| `AscendExtractHiddenStatesProposer` | `vllm_ascend/spec_decode/extract_hidden_states_proposer.py` | NPU 适配 proposer：SP/DP padding、ACL graph dummy_run、discard 索引处理 |
| `ExtractHiddenStatesProposer` | 上游 `vllm/v1/spec_decode/extract_hidden_states.py` | propose / load_model / cudagraph 核心逻辑 |
| `ExtractHiddenStatesModel` | 上游 `vllm/model_executor/models/extract_hidden_states.py` | 仅含一层 `CacheOnlyAttentionLayer` 的草稿模型 |
| `ExampleHiddenStatesConnector` | 上游 KV connector | 异步落盘 |
| `get_spec_decode_method()` | `vllm_ascend/spec_decode/__init__.py` | 工厂注册（约 48-49 行） |
| `NPUModelRunner` 集成点 | `vllm_ascend/worker/model_runner_v1.py` | drafter 装配、propose 路径、KV 分配、cudagraph key、aux 层配置 |
| 混合模型 patch | `vllm_ascend/patch/platform/patch_mamba_config.py` | 该 connector + 混合模型时跳过 mamba 强制 align 模式（约 125-136 行） |
| MoE 误判保护 | `vllm_ascend/utils.py` | 防止 MoE 目标模型在 DP 下死锁（约 895-901 行） |

### Ascend proposer 的三处关键 override

`vllm_ascend/spec_decode/extract_hidden_states_proposer.py` 中：

1. **`_determine_batch_execution_and_padding`**（约 42-112 行）：替换上游 `coordinate_batch_across_dp`。
   上游实现会向 DP 的 `cpu_group` 投递一个**形状不同**的张量，破坏 Ascend 共享 DP cpu_group 上的 gloo 集合通信。
   Ascend 版改为先做 `runner._pad_for_sequence_parallelism`（SP padding），再复用 `runner._sync_metadata_across_dp(is_draft_model=True)` 与主前向保持一致的集合通信形状。
2. **`dummy_run`**（约 114-156 行）：ACL graph 捕获签名 + 在空闲 DP rank 上**强制**发起同样的 draft DP sync，避免 DP cpu_group 集合通信错位导致死锁。
3. **`prepare_next_token_ids_padded`**（约 158-198 行）：Ascend 采用 `discard_request_indices` + 计数的模式（而非 GPU 的 bool mask），构造 discard mask 并选择 sampled / backup token。

### v1 model runner 集成点（仅 v1）

`vllm_ascend/worker/model_runner_v1.py` 中（行号为参考）：

- **drafter 装配**（约 623-625 行）：`method == "extract_hidden_states"` 时 `use_aux_hidden_state_outputs = True`。
- **propose 路径**（约 1754-1782 行）：调用 `drafter.propose(...)` + `drafter.prepare_next_token_ids_padded(...)`；若 `aux_hidden_states` 为空则抛错。
- **aux 层配置**（约 3744-3761 行）：要求目标模型实现 EAGLE3 接口（`supports_eagle3`），否则报错；随后 `set_aux_hidden_state_layers(...)`。
- **KV cache 分配**（约 4406-4411 行）：`cache_only_layers` / `HiddenStateCacheSpec` 用**单张量**（不拆 K/V）。
- **KV spec 发现**（约 4887-4907 行）：为混合模型（如 Qwen3.5）重建可 pickle 的 `HiddenStateCacheSpec`。

## 4. 测试

### E2E

`tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py`，单个参数化用例 `test_extract_hidden_states`，含三个 case：

| Case | 模型 | 权重 | 模式 | 校验点 |
| --- | --- | --- | --- | --- |
| `dense_eager` | Qwen3-8B | 真实 | eager | hidden states 非零 |
| `dense_aclgraph` | Qwen3-8B | 真实 | ACL graph | hidden states 非零 |
| `hybrid_dummy_eager` | Qwen3.5-0.8B（GatedDeltaNet + full_attention） | dummy | eager | 形状 + `token_ids` 回环 |

期望形状：`(len(prompt_token_ids), len(layer_ids), hidden_size)`。
CI 入口：`.github/workflows/scripts/test_config.yaml` 中的 `spec_decode_extract_hidden_states` job。

### 单元测试

`tests/ut/spec_decode/test_extract_hidden_states_proposer.py`，覆盖初始化、`dummy_run`、DP sync 防死锁回归、discard 索引模式、SP padding / 无 `coordinate_batch_across_dp` 等。
另有 `tests/ut/test_utils.py` 中关于 MoE 误判保护的用例。

## 5. 环境变量

本特性在 vLLM Ascend 侧**无专用环境变量**。E2E 测试仅设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。

## 6. 已知约束、坑与 TODO

### 功能性约束（多数来自上游）

- `num_speculative_tokens` 必须为 `1`。
- **不支持** `disable_padded_drafter_batch=True`（上游会抛 `ValueError`）。
- `eagle_aux_hidden_state_layer_ids` 必填。
- 目标模型必须实现 **EAGLE3 辅助 hidden state 接口**（`supports_eagle3` / `set_aux_hidden_state_layers`）。
- `kv_connector` 必须是 `ExampleHiddenStatesConnector`。
- 通常配合 `max_tokens=1`，输出文本为次要产物。
- **仅支持 v1 model runner**；v2 缺少 `_pad_for_sequence_parallelism` / `_sync_metadata_across_dp`，会显式抛 `NotImplementedError`。

### Ascend 特有的坑

- **DP > 1**：必须走 `runner._sync_metadata_across_dp(is_draft_model=True)`，**不能**用上游 `coordinate_batch_across_dp`（gloo 形状不匹配会死锁）。
- **序列并行（SP）**：DP 分发前必须先 `_pad_for_sequence_parallelism`。
- **MoE 目标模型**：草稿 HF config 会继承 MoE 的 key，需要保证 `is_drafter_moe_model()` 返回 `False`，否则会因多余的 DP all_reduce 死锁（见 `utils.py` 的保护逻辑）。
- **混合模型（Qwen3.5）**：需要特殊的 `HiddenStateCacheSpec` 与 page size padded stride 逻辑；mamba 的 align 模式通过 `patch_mamba_config.py` 跳过。
- `ExampleHiddenStatesConnector` 上游使用 `torch.cuda.Stream` / `Event`，本仓库没有 NPU 专门的 connector patch，依赖上游/设备兼容性，升级上游时需关注。

### TODO（继承自上游）

- 上游 proposer 暂不支持 DBO ubatching（`assert not should_ubatch`）。

## 7. 文件索引

| 路径 | 关注区间 |
| --- | --- |
| `vllm_ascend/spec_decode/extract_hidden_states_proposer.py` | 全文（25-198 行） |
| `vllm_ascend/spec_decode/__init__.py` | 工厂注册（48-49 行） |
| `vllm_ascend/worker/model_runner_v1.py` | 623-625、1754-1782、3744-3761、4406-4411、4887-4907 |
| `vllm_ascend/utils.py` | 895-901（MoE 误判保护） |
| `vllm_ascend/patch/platform/patch_mamba_config.py` | 125-136 |
| `docs/source/user_guide/feature_guide/speculative_decoding.md` | 216-288（对外使用文档） |
| `tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py` | 全文 |
| `tests/ut/spec_decode/test_extract_hidden_states_proposer.py` | 全文 |

### 相关提交（便于追溯）

- `b3196cc8` [SpecDecode][Feature] Implement AscendExtractHiddenStatesProposer for speculative decoding (#8799)
- `1eaaf34e` [BugFix] Fix multi-DP deadlock for extract_hidden_states proposer on MoE targets (#9689)
