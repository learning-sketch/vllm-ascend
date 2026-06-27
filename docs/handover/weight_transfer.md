# 交接文档：权重传输（Weight Transfer）

> 关键词：RL/RLHF、在线权重同步、HCCL 后端、NPU IPC 后端、packed 传输、dev HTTP 控制面。

## 1. 这是什么 / 为什么需要

权重传输让**训练进程（trainer）在推理服务不重启的前提下，把最新模型权重在线灌入正在运行的 vLLM 推理 Worker**。
主要服务 **RL/RLHF** 闭环：训练更新权重 → 推送给推理引擎 → 生成 rollouts → 再训练，如此循环。

典型 RLHF 流程（以 dev HTTP 控制面为例）：

1. 用 `--load-format dummy` 启动推理服务并开启权重传输；
2. 用 dummy 权重先生成一遍 baseline（通常是乱码）；
3. `/pause` 暂停生成；
4. `/init_weight_transfer_engine` 初始化传输（建立 HCCL group 或准备 NPU IPC）；
5. `/start_weight_update` 准备模型做原地重载；
6. `/update_weights` 真正搬运权重（HCCL broadcast 或 IPC handle）；
7. `/finish_weight_update` 收尾重载；
8. `/resume` 用真实权重恢复生成。

示例脚本：`examples/rl/rlhf_http_hccl.py`、`examples/rl/rlhf_http_npu_ipc.py`。

> 这与 RFork/NetLoader 的权重加载、EPLB 专家权重迁移（`eplb_device_transfer_loader.py`）、layer-sharding 的异步 broadcast 都**不是**同一套机制。

实现形态：vLLM Ascend 提供两个 **Ascend 专用后端**，插入上游 vLLM 的权重传输框架（`WeightTransferEngine` / `WeightTransferEngineFactory` / dev HTTP 端点）。

## 2. 两种传输后端

### HCCL 后端（`HCCLWeightTransferEngine`）

文件：`vllm_ascend/distributed/weight_transfer/hccl_engine.py`

| 维度 | 说明 |
| --- | --- |
| 机制 | 通过 `PyHcclCommunicator` 在 vLLM 的 `StatelessProcessGroup` 上做 HCCL broadcast |
| 拓扑 | trainer（rank 0）+ 所有推理 Worker（rank ≥ `rank_offset`）组成一个 HCCL group |
| 是否同卡 | **不要求**同卡，trainer 与推理可在**不同 NPU**（如 NPU0 推理、NPU1 训练） |
| 初始化入参 | `master_address`、`master_port`、`rank_offset`、`world_size` |
| 数据流 | trainer broadcast，Worker 通过 `group.broadcast()` 接收 |
| packed 模式 | 把大量小张量打包进 ~1GB 的 uint8 buffer，双/三缓冲流水 |
| 用户侧 backend 字符串 | `"nccl"`（由 patch 映射到 HCCL，Ascend 上无 NCCL） |

**适用**：多卡 / 多机；trainer 与推理不在同一物理卡；模型较大、同卡放不下两份的场景。

### NPU IPC 后端（`NPUIPCWeightTransferEngine`）

文件：`vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py`

| 维度 | 说明 |
| --- | --- |
| 机制 | Ascend NPU IPC：`torch.multiprocessing.reductions.reduce_tensor` / `torch_npu...rebuild_npu_tensor` |
| 拓扑 | trainer 与推理 Worker 必须共享**同一块物理 NPU** |
| 是否同卡 | **要求**同卡，用 `{host_ip}-{physical_chip_id}` 作为 UUID 匹配 |
| 初始化 | no-op（`init_transfer_engine` 为 `pass`） |
| 数据流 | trainer 生成 IPC handle，元数据（+ pickle 的 handle，经 HTTP）发给 Worker，Worker 进程内 rebuild |
| 传输到服务的方式 | `send_mode`：`"http"` / `"ray"` / 自定义 callable |
| packed 模式 | 单个可复用 NPU buffer，分块发送以控内存 |
| 用户侧 backend 字符串 | `"ipc"`（由 patch 映射到 NPU IPC） |

**适用**：单卡 / 同卡共置 trainer + 推理；省去 HCCL 建链；当显存能放下两份模型时（调低 `--gpu-memory-utilization`）可能更快。

### 对照速查

| | HCCL（`"nccl"`） | NPU IPC（`"ipc"`） |
| --- | --- | --- |
| 典型最少 NPU 数 | 2（1 推理 + 1 训练） | 1（共享） |
| 是否需要 HCCL 建链 | 是 | 否 |
| HTTP init 负载 | `{master_address, master_port, rank_offset, world_size}` | `{}` |
| 权重数据是否走 HTTP | 否，仅元数据 | 是，元数据 + pickle 的 IPC handle |
| 多 rank trainer | rank 0 broadcast | `all_gather_object` 合并各 rank 的 IPC handle |
| 并发要求 | 服务端 `/update_weights` 必须与 trainer 的 HCCL send **并发**执行 | 同步 POST，无集合通信死锁 |

## 3. 如何开启与配置

### CLI（主入口）

```bash
# HCCL
vllm serve MODEL --enforce-eager --load-format dummy \
  --weight-transfer-config '{"backend": "nccl"}'

# NPU IPC
vllm serve MODEL --enforce-eager --load-format dummy \
  --gpu-memory-utilization 0.5 \
  --weight-transfer-config '{"backend": "ipc"}'
```

`"nccl"` / `"ipc"` 是**上游 vLLM 的字面量**，由 Ascend patch 重映射（见 §5）。

### 相关环境变量（均非 `vllm_ascend/envs.py` 定义）

| 变量 | 用于 | 作用 |
| --- | --- | --- |
| `VLLM_SERVER_DEV_MODE=1` | HTTP 控制面 | 注册 dev 端点：`/init_weight_transfer_engine`、`/update_weights`、`/start_weight_update`、`/finish_weight_update`、`/pause`、`/resume`、`/get_world_size` |
| `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | NPU IPC 走 HTTP | 允许服务端 unpickle `ipc_handles_pickled` |
| `VLLM_ASCEND_ENABLE_NZ=0` | RL 权重更新 | FRACTAL_NZ 会破坏 RL 权重精度，Worker 端强制检查 |
| `ASCEND_RT_VISIBLE_DEVICES` | NPU IPC UUID 对齐 | 逻辑→物理 chip id 映射 |
| `WEIGHT_TRANSFER_TEST_MODEL` | 仅 E2E 测试 | 指定真实 ckpt 路径，否则用随机权重 |

> `HCCL_SO_PATH`（`vllm_ascend/envs.py` 约 59-61 行）影响 `PyHcclCommunicator` 通用行为，并非权重传输专属配置。

### 推荐服务端 flag

- `--enforce-eager`
- `--load-format dummy`（用占位权重启动）
- `--gpu-memory-utilization 0.5`（NPU IPC，给 trainer 留显存）
- `--trust-remote-code`（视模型而定）

## 4. 架构与调用链

### 关键文件

| 路径 | 角色 |
| --- | --- |
| `vllm_ascend/distributed/weight_transfer/hccl_engine.py` | HCCL engine + trainer 辅助函数 |
| `vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py` | NPU IPC engine + trainer 辅助函数 |
| `vllm_ascend/distributed/weight_transfer/packed_tensor.py` | packed broadcast（HCCL）+ packed IPC 分块 |
| `vllm_ascend/distributed/weight_transfer/__init__.py` | 插件注册（`hccl`、`npu_ipc`） |
| `vllm_ascend/patch/platform/patch_weight_transfer_engine.py` | 启动时把 `"nccl"`→HCCL、`"ipc"`→NPU IPC |
| `vllm_ascend/worker/worker.py` | Worker 生命周期：engine 创建、HTTP handler 落点、权重更新状态机 |
| `vllm_ascend/distributed/device_communicators/pyhccl.py` | HCCL broadcast 实现 |
| `examples/rl/rlhf_http_*.py` | 端到端用法示例 |

> 注意：`model_runner_v1.py` 里的 `packed_tensor`（约 681-691 行）是给 DP token / cudagraph 同步用的，**与权重传输无关**。权重传输只集成在 `worker.py`。

### 两条注册路径

1. **平台 patch**（对应 CLI 的 `"nccl"` / `"ipc"`），在 `NPUPlatform.pre_register_and_update()` 阶段、`create_engine()` 之前生效：

   ```python
   # vllm_ascend/patch/platform/patch_weight_transfer_engine.py
   WeightTransferEngineFactory._registry["nccl"] = lambda: HCCLWeightTransferEngine
   WeightTransferEngineFactory._registry["ipc"] = _load_npu_ipc_engine
   ```

2. **插件注册**（`"hccl"` / `"npu_ipc"` 名字），由 `vllm_ascend:register_connector` 入口调用：

   ```python
   # vllm_ascend/distributed/weight_transfer/__init__.py
   WeightTransferEngineFactory.register_engine("hccl", "...hccl_engine", "HCCLWeightTransferEngine")
   WeightTransferEngineFactory.register_engine("npu_ipc", "...npu_ipc_engine", "NPUIPCWeightTransferEngine")
   ```

   **用户侧 CLI 用的是 patch 别名 `"nccl"`/`"ipc"`，不是 `"hccl"`/`"npu_ipc"`**，这点容易混淆。

### Worker 端 engine 创建

```python
# vllm_ascend/worker/worker.py，load_model() 内、模型加载之后
if self.vllm_config.weight_transfer_config is not None:
    self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(
        self.vllm_config.weight_transfer_config,
        self.vllm_config.parallel_config,
        self.model_runner.get_model(),
    )
```

### Worker 权重更新状态机（`worker.py`，约 305-376 行）

- `start_weight_update(is_checkpoint_format=True)`：先 `_check_nz_disabled()`，再 `initialize_layerwise_reload(model)`，置 `_weight_update_active = True`。
- `update_weights(update_info)`：未先 `start_weight_update` 会抛错；随后 `weight_transfer_engine.receive_weights(...)` + `torch.npu.synchronize()`。
- `finish_weight_update()`：必要时 `finalize_layerwise_reload(...)`，复位 `_weight_update_active`。
- `is_checkpoint_format=True`（默认）走 `model.load_weights` + layerwise reload；`False` 走 `param.copy_(weight)`（`get_parameter(name)` 直拷）。

### HTTP 端点 → Worker 方法映射

| HTTP 端点 | Worker 方法 |
| --- | --- |
| `/init_weight_transfer_engine` | `init_weight_transfer_engine(init_info)` |
| `/start_weight_update` | `start_weight_update(is_checkpoint_format)` |
| `/update_weights` | `update_weights(update_info)` |
| `/finish_weight_update` | `finish_weight_update()` |
| `/pause`、`/resume` | 上游 pause/resume |
| `/get_world_size` | HCCL 示例用来确定 HCCL group 规模 |

### HCCL rank 分配（`hccl_engine.py` 约 143-151 行）

```python
dp_rank = self.parallel_config.data_parallel_index
world_size_per_dp = self.parallel_config.world_size  # TP * PP
rank_within_dp = self.parallel_config.rank
worker_rank = dp_rank * world_size_per_dp + rank_within_dp
rank = worker_rank + init_info.rank_offset
```

### `packed_tensor.py` 的作用

批量高效传输的共用工具：

| 函数 | 使用方 | 作用 |
| --- | --- | --- |
| `packed_broadcast_producer` | HCCL trainer | 打包成 uint8 buffer，按流水做 HCCL broadcast |
| `packed_broadcast_consumer` | HCCL worker | 接收、解包、增量 `load_weights` |
| `packed_npu_ipc_producer` | IPC trainer | 打进一个可复用 IPC buffer，分块 yield |
| `packed_npu_ipc_consumer` | IPC worker | 重建 IPC buffer、切分、**clone**（因为 producer 复用 buffer） |

默认值：`DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1GB`，`DEFAULT_PACKED_NUM_BUFFERS = 2`。
HCCL packed 用跨 NPU stream 的双/三缓冲；NPU IPC packed 用单块可复用 IPC 分配，逐块拷入后发送。

## 5. 入口与上游集成

- **patch（`patch_weight_transfer_engine.py`）**：目标是 `WeightTransferEngineFactory._registry`。
  原因：上游 `WeightTransferConfig.backend` 是 `Literal["nccl", "ipc"]`，要新增 `"hccl"`/`"npu_ipc"` 得改 pydantic schema。
  `_load_npu_ipc_engine` 做**懒加载**，避免在不用 IPC 时就触发上游 `import ray`。
  未来上游放开 Literal / 延后 ray import 后可移除此 patch（见 `patch/__init__.py` 的 TODO）。
- **RLHF 示例**：
    - HCCL（`rlhf_http_hccl.py`）：trainer 放在 `npu:{inference_world_size}`（推理之后的下一张卡），用线程并发跑 init + update（HCCL 集合通信两端都阻塞），`packed=True`。
    - NPU IPC（`rlhf_http_npu_ipc.py`）：trainer 与服务同在 `npu:0`，设 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`，`trainer_send_weights(send_mode="http")` POST 到 `/update_weights`。

## 6. 测试

### E2E（需 NPU 硬件）

| 测试 | 路径 | 硬件 | 后端 |
| --- | --- | --- | --- |
| NPU IPC | `tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py` | ≥1 NPU | `"ipc"` |
| HCCL | `tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py` | ≥2 NPU | `"nccl"` |

两者都：用 dummy 权重启 `vllm serve`（`RemoteOpenAIServer`）；更新前后各生成一次并断言输出不同；默认用 config 随机权重（免下载），可用 `WEIGHT_TRANSFER_TEST_MODEL` 指定真实 ckpt；需要 `VLLM_SERVER_DEV_MODE=1`。
CI：`.github/workflows/scripts/test_config.yaml`（约 232-239 行）的可选 `weight_transfer` 组。

### 单元测试

- `tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py`：IPC handle 格式（仅 args，不是 `(func, args)`）、`__init__(model=...)`、`rebuild_npu_tensor` 设备 index 覆盖。
- `tests/ut/worker/a2/test_worker_v1.py`（`TestNPUWorkerWeightUpdate`，约 1338-1520 行）：Worker 状态机、NZ 拒绝、checkpoint vs kernel 格式、shutdown。

HCCL engine 暂无独立 UT，由 2 卡 E2E 覆盖。

## 7. 已知约束、坑与 TODO

### 硬性失败点

1. **FRACTAL_NZ 与 RL 权重更新不兼容**：`start_weight_update` 在 `VLLM_ASCEND_ENABLE_NZ` 为真时直接抛错（`worker.py` 的 `_check_nz_disabled`，约 297-303 行），必须 `VLLM_ASCEND_ENABLE_NZ=0`。
2. **NPU IPC 必须同卡共置**：UUID 不匹配会抛 `ValueError`（`npu_ipc_engine.py` 约 196-201 行）。
3. **NPU IPC 走 HTTP 的 pickle 安全**：需显式 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`。
4. **packed buffer 大小**：单个张量超过 `buffer_size_bytes` 会在 `packed_npu_ipc_producer` 抛错（约 234-238 行）。
5. **状态机顺序**：必须先 `start_weight_update` 再 `update_weights`；未 `finish_weight_update` 不可重入。

### 运行期坑

- **HCCL 死锁**：服务端 `/update_weights` 必须与 trainer 的 `trainer_send_weights` **并发**（示例/测试里用后台线程）。
- **NPU IPC 显存**：同卡放两份模型，调低 `--gpu-memory-utilization`；trainer 张量在 send 完成前不能释放（`_send_unpacked` 的 `weight_refs`）。
- **packed IPC consumer 必须 clone**：因为 producer 复用同一块 IPC buffer。
- **多 rank trainer（IPC）**：所有 rank 都要调 `trainer_send_weights`，handle 经 `all_gather_object` 合并，仅 rank 0 发 HTTP。
- **逻辑 vs 物理设备 index**：接收端会用本地设备 index 覆盖 IPC args 的 index 6（`npu_ipc_engine.py` 约 205-209 行）。
- **vLLM 版本漂移**：E2E 会探测生命周期端点；旧版本可能缺 `/start_weight_update` / `/finish_weight_update`，HCCL 测试有回退分支。

### TODO（代码注释中）

- 上游接受自定义 backend 字符串 / 提供扩展点后，移除工厂 patch。
- 上游不再在 `ipc_engine` 顶层 `import ray` 后，移除懒加载。
- 插件注册的是 `"hccl"`/`"npu_ipc"` 而 CLI 用 `"nccl"`/`"ipc"`，文档上需说明清楚以免误用。

## 8. 端到端命令速查

```bash
# HCCL（2 卡）
VLLM_SERVER_DEV_MODE=1 VLLM_ASCEND_ENABLE_NZ=0 \
  vllm serve Qwen/Qwen3-0.6B --enforce-eager --load-format dummy \
  --weight-transfer-config '{"backend": "nccl"}'
python examples/rl/rlhf_http_hccl.py

# NPU IPC（1 卡共置）
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_ASCEND_ENABLE_NZ=0 \
  vllm serve Qwen/Qwen3-0.6B --enforce-eager --load-format dummy \
  --weight-transfer-config '{"backend": "ipc"}' \
  --gpu-memory-utilization 0.5
python examples/rl/rlhf_http_npu_ipc.py
```

## 9. 文件索引与相关提交

| 路径 | 关注点 |
| --- | --- |
| `vllm_ascend/distributed/weight_transfer/hccl_engine.py` | engine + trainer helper、rank 分配（143-151） |
| `vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py` | engine、UUID（74-99）、handle 校验（196-209） |
| `vllm_ascend/distributed/weight_transfer/packed_tensor.py` | packed 工具、默认值（14-15） |
| `vllm_ascend/distributed/weight_transfer/__init__.py` | 插件注册（21-32） |
| `vllm_ascend/patch/platform/patch_weight_transfer_engine.py` | backend 重映射（72-73） |
| `vllm_ascend/worker/worker.py` | engine 创建（701-711）、状态机（305-376）、NZ 检查（297-303） |
| `tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py` | NPU IPC E2E |
| `tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py` | HCCL E2E |
| `tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py` | IPC UT |

相关提交：

- `324dc45f` [Feature] WeightTransfer: Add HCCLWeightTransferEngine backend for Ascend NPU (#9152)
- `6d428b6f` [Feature] WeightTransfer: Add NPUIPCWeightTransferEngine backend for Ascend NPU (#10592)
- `5ca762a7` [BugFix][WeightTransfer] Fix NPU IPC engine init and align IPC handle with upstream (#10996)
