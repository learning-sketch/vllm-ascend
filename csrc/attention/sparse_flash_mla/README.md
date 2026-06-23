# SparseFlashMla

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        | ×  |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        | √  |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        | √  |
|<term>Atlas 200I/500 A2推理系列产品</term>                    | ×  |
|<term>Atlas 推理系列产品</term>                                | ×  |
|<term>Atlas 训练系列产品</term>                                | ×  |

## 功能说明
- 算子功能：`SparseFlashMla`算子旨在完成以下公式描述的Attention计算，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于ori_kv、cmp_kv以及cmp_ratio等入参控制的实际参与计算的 $KV$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>q</td>
      <td>输入</td>
      <td>对应公式中的$Q$。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_kv</td>
      <td>可选输入</td>
      <td>对应公式中的$\tilde{K}和\tilde{V}$的一部分，为原始不经压缩的KV。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_kv</td>
      <td>可选输入</td>
      <td>对应公式中的$\tilde{K}和\tilde{V}$的一部分，为经过压缩的KV。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_sparse_indices</td>
      <td>可选输入</td>
      <td>代表离散取oriKvCache的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_sparse_indices</td>
      <td>可选输入</td>
      <td>代表离散取cmpKvCache的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention中oriKvCache存储使用的block映射表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention中cmpKvCache存储使用的block映射表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中`q`的有效token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中`ori_kv`的有效token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中`cmp_kv`的有效token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中`q`实际参与运算的token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中`ori_kv`实际参与运算的token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>可选输入</td>
      <td>占位输入，当前未使用。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>可选输入</td>
      <td>占位输入，当前未使用。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>可选输入</td>
      <td>占位输入，当前未使用。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>可选输入</td>
      <td>占位输入，当前未使用。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinks</td>
      <td>可选输入</td>
      <td>注意力下沉tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>可选输入</td>
      <td>aicpu算子（sparse_flash_mla_metadata）的分核结果。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_scale</td>
      <td>可选属性</td>
      <td>代表缩放系数，对应公式中的$\text{softmax\_scale}$，默认值为None。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示对`ori_kv`的压缩率，仅支持输入4或128，默认值为None。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算的mask模式，仅支持输入默认值4。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示`q`和`cmp_kv`计算的mask模式，仅支持输入默认值3。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算中q对过去token计算的数量，仅支持输入默认值127。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算中q对未来token计算的数量，仅支持输入默认值0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>用于标识输入`q`的数据排布格式，支持输入"TND"和"BSND"，默认值为"BSND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>用于标识输入`ori_kv`和`cmp_kv`的数据排布格式，支持输入"PA_BBND"和"BSND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>可选属性</td>
      <td>表示是否返回`softmax_lse`。True表示返回，False表示不返回，默认值为False。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_lse</td>
      <td>输出</td>
      <td>返回的`softmax_lse`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明
- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 该接口当前支持三种计算场景：场景一，仅传入`ori_kv`时为Sliding Window Attention计算；场景二，传入`ori_kv`及`cmp_kv`时为Sliding Window Attention + Compressed Attention计算；场景三，传入`ori_kv`、`cmp_kv`及`cmp_sparse_indices`时为Sliding Window Attention + Sparse Compressed Attention计算。

- 当`layout_q`为TND时，功能使用限制如下：
  - `q`的shape需要为[T1,N1,D]，其中N1仅支持64。
  - `ori_sparse_indices`的shape需要为[Q\_T, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的token数，K1仅支持512。
  - `cmp_sparse_indices`的shape需要为[Q\_T, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数，K2仅支持512。
  - `cu_seqlens_q`必须传入，输入维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值。

- 当`layout_q`为BSND时，功能使用限制如下：
  - `q`的shape需要为[B, Q\_S,N1,D]，其中N1仅支持64。
  - `ori_sparse_indices`的shape需要为[B, Q\_S, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的token数，K1仅支持512。
  - `cmp_sparse_indices`的shape需要为[B, Q\_S, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数，K2仅支持512。

- PageAttention场景下，功能使用限制如下：
  - `ori_kv`和`cmp_kv`的shape分别为[ori\_block\_num, ori\_block\_size, KV\_N, D]和[cmp\_block\_num, cmp\_block\_size, KV\_N, D]，其中ori\_block\_num和cmp\_block\_num为PageAttention时block总数，ori\_block\_size和cmp\_block\_size为一个block的token数，ori\_block\_size和cmp\_block\_size取值为16的倍数，最大支持1024，KV_N仅支持1。
  - `ori_block_table`和`cmp_block_table`的shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2和S3对应的block数量，即S2\_max / block\_size和S3\_max / block\_size向上取整。
- `metadata`为算子实际需要使用的分核结果，目前该参数必传，shape大小固定为[1024]。
- `layout_kv`仅支持输入"PA_BBND"和"BSND"。
  - 当输入为PA_BBND时，设置`cu_seqlens_ori_kv`和`cu_seqlens_cmp_kv`无效。
  - 当输入为BSND时，`ori_kv`和`cmp_kv`的layout都必须为BSND，ori_kv的shape为[B, S2, N2,D]，cmp_kv的shape为[B, S3, N2,D]。
- 目前暂不支持返回`softmax_lse`，`return_softmax_lse`仅支持输入False，返回值`softmax_lse`为无效值。
- 目前暂不支持指定`q`中参与运算的token数，因此设置`seqused_q`无效。
- 目前暂不支持对`ori_kv`进行稀疏计算，因此设置`ori_sparse_indices`无效。
- 目前所有输入不支持传入空tensor。
- `q`、`ori_kv`、`cmp_kv`数据排布格式支持从多种维度解读，B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
- Q\_S和S1表示q shape中的S，S2表示ori_kv shape中的S，S3表示cmp_kv shape中的S；Q\_N和N1表示num\_q\_heads，KV\_N和N2表示num\_ori_kv\_heads和num\_cmp_kv\_heads；Q\_T和T1表示q shape中的输入样本序列长度的累加和。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnnSparseFlashMla](./examples/test_aclnn_sparse_flash_mla.cpp) | 通过[aclnnSparseFlashMla](./docs/aclnnSparseFlashMla.md)调用SparseFlashMla算子 |
