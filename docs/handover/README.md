# vLLM Ascend 特性交接文档

本目录用于沉淀 vLLM Ascend 关键特性的**工程交接材料（交接文档）**，面向后续接手维护、排障与二次开发的同学。
区别于 `docs/source/` 下面向最终用户的使用文档，本目录侧重于：

- 特性**为什么存在**（背景与使用场景）；
- **代码落在哪里**（关键文件、类、调用链）；
- **怎么开启与配置**（环境变量、配置项、API）；
- **怎么测试**（UT / E2E、CI 入口）；
- **已知的坑、约束与 TODO**。

> 说明：本目录下的文档是内部交接材料，未接入 Sphinx 文档站点（`docs/source` 的 toctree），
> 因此不会影响对外发布的文档构建。

## 文档索引

| 特性 | 文档 | 一句话概述 |
| --- | --- | --- |
| Extract Hidden States | [extract_hidden_states.md](./extract_hidden_states.md) | 复用投机解码（spec decode）管线，从目标模型指定层抽取 hidden states 并落盘，用于 EAGLE 类草稿模型的训练数据采集。 |
| 权重传输（Weight Transfer） | [weight_transfer.md](./weight_transfer.md) | RL/RLHF 场景下，训练进程在推理服务**不重启**的前提下，把最新权重在线灌入推理 Worker。提供 HCCL 与 NPU IPC 两种传输后端。 |
| Token in Token out | [token_in_token_out.md](./token_in_token_out.md) | 直接以 token_ids 作为输入、直接返回 token_ids 的推理路径（`/inference/v1/generate`），避免文本侧的反复 tokenize/detokenize，主要服务 RL 与 Disaggregated 协调器。 |

## 通用背景：vLLM Ascend 是插件

vLLM Ascend 是 vLLM 的**硬件后端插件**，通过 vLLM 的可插拔硬件接口接入，本身原则上不直接新增模型文件。
绝大多数“特性”是以下三种形态之一：

1. **继承扩展**：例如 `NPUModelRunner(GPUModelRunner)`、`AscendXxxProposer(UpstreamXxxProposer)`；
2. **Patch 上游**：`vllm_ascend/patch/platform/` 与 `vllm_ascend/patch/worker/`；
3. **上游能力 + NPU 适配**：许多协议/接口在上游 vLLM 中定义，Ascend 侧只做 NPU 代码路径的适配与验证。

理解这三种形态，有助于快速判断“这个特性的主体逻辑到底在上游 vLLM 还是在 vLLM Ascend”。

## 环境与约定

- 环境变量统一定义在 `vllm_ascend/envs.py` 的 `env_variables` 字典，命名遵循 `VLLM_ASCEND_*`，详见 [AGENTS.md](../../AGENTS.md)。
- 提交需 sign-off：`git commit -s`；推送前请跑 `bash format.sh ci`（含 markdownlint，对 Markdown 同样生效）。
- 各特性文档末尾均附“文件索引”，标注关键文件与行号区间，便于快速跳转源码。
