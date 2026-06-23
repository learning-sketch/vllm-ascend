from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import (
    A5DeviceAdaptor,
    BaseDeviceAdaptor,
    _bf16_sparse_flash_mla,
    _bf16_sparse_flash_mla_metadata,
)


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


@pytest.mark.parametrize("use_bf16", [False, True])
def test_a5_sparse_attn_op_selection_by_kv_dtype(use_bf16):
    mock_torch = mock.MagicMock()
    ops = mock_torch.ops._C_ascend
    with (
        mock.patch("vllm_ascend.device.device_op.dsv4_use_kv_bf16", return_value=use_bf16),
        mock.patch("vllm_ascend.device.device_op.torch", mock_torch),
    ):
        attn_op = A5DeviceAdaptor.get_dsa_sparse_attn_op()
        metadata_op = A5DeviceAdaptor.get_dsa_sparse_attn_metadata_op()

    if use_bf16:
        # BF16 returns kwarg-renaming wrappers around the sparse_flash_mla ops
        # (the call sites use the FP8 shared-KV kwarg names).
        assert attn_op is _bf16_sparse_flash_mla
        assert metadata_op is _bf16_sparse_flash_mla_metadata
    else:
        assert attn_op is ops.npu_kv_quant_sparse_attn_sharedkv
        assert metadata_op is ops.npu_kv_quant_sparse_attn_sharedkv_metadata


def test_a5_bf16_sparse_flash_mla_wrappers_rename_kwargs():
    mock_torch = mock.MagicMock()
    ops = mock_torch.ops._C_ascend
    with mock.patch("vllm_ascend.device.device_op.torch", mock_torch):
        _bf16_sparse_flash_mla("Q", seqused_kv="S", sinks="X", metadata="M")
        _bf16_sparse_flash_mla_metadata(
            num_heads_q=64, seqused_kv="S", max_seqlen_kv="MK", cmp_ratio=0
        )

    # attn op: q stays positional, seqused_kv -> seqused_ori_kv, rest untouched.
    ops.npu_sparse_flash_mla.assert_called_once_with("Q", seqused_ori_kv="S", sinks="X", metadata="M")
    attn_kwargs = ops.npu_sparse_flash_mla.call_args.kwargs
    assert "seqused_kv" not in attn_kwargs

    # metadata op: seqused_kv -> seqused_ori_kv, max_seqlen_kv -> max_seqlen_ori_kv.
    ops.npu_sparse_flash_mla_metadata.assert_called_once_with(
        num_heads_q=64, seqused_ori_kv="S", max_seqlen_ori_kv="MK", cmp_ratio=0
    )
    meta_kwargs = ops.npu_sparse_flash_mla_metadata.call_args.kwargs
    assert "seqused_kv" not in meta_kwargs
    assert "max_seqlen_kv" not in meta_kwargs


@pytest.mark.parametrize("use_bf16", [False, True])
def test_a5_sparse_attn_kwargs_and_layout_by_kv_dtype(use_bf16):
    with mock.patch("vllm_ascend.device.device_op.dsv4_use_kv_bf16", return_value=use_bf16):
        base_kwargs = A5DeviceAdaptor.get_dsa_sparse_attn_base_kwargs()
        metadata_kwargs = A5DeviceAdaptor.get_dsa_sparse_attn_metadata_kwargs("npu:0")
        layout = A5DeviceAdaptor.get_dsa_kv_layout()
        swa_only_cmp_ratio = A5DeviceAdaptor.get_dsa_swa_only_cmp_ratio()

    if use_bf16:
        # sparse_flash_mla drops the FP8-quant-only attributes.
        assert base_kwargs == {}
        assert "kv_quant_mode" not in metadata_kwargs
        assert metadata_kwargs == {"device": "npu:0"}
        assert layout == "PA_BBND"
        # SWA-only scenario passes 0 (no compression) for sparse_flash_mla.
        assert swa_only_cmp_ratio == 0
    else:
        assert base_kwargs == {"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64}
        assert metadata_kwargs == {"kv_quant_mode": 1}
        assert layout == "PA_ND"
        assert swa_only_cmp_ratio == 1


def test_a5_bf16_kv_scatter_delegates_to_plain_scatter():
    cache = object()
    x = object()
    slot_mapping = object()
    with (
        mock.patch("vllm_ascend.device.device_op.dsv4_use_kv_bf16", return_value=True),
        mock.patch.object(BaseDeviceAdaptor, "dsa_kv_compress_scatter") as mock_scatter,
    ):
        A5DeviceAdaptor.dsa_kv_compress_scatter(cache, x, slot_mapping)
    mock_scatter.assert_called_once_with(cache, x, slot_mapping)


def test_a5_bf16_slot_mapping_uses_2d_format():
    slot_mapping = torch.tensor([5, 18, 33], dtype=torch.int32)
    with mock.patch("vllm_ascend.device.device_op.dsv4_use_kv_bf16", return_value=True):
        formatted = A5DeviceAdaptor.format_dsa_slot_mapping(slot_mapping, block_size=16)
    # 2D [block_idx, offset] like the A2/A3 path.
    assert formatted.shape == (3, 2)
    torch.testing.assert_close(formatted[:, 0], torch.tensor([0, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(formatted[:, 1], torch.tensor([5, 2, 1], dtype=torch.int32))

    with mock.patch("vllm_ascend.device.device_op.dsv4_use_kv_bf16", return_value=False):
        passthrough = A5DeviceAdaptor.format_dsa_slot_mapping(slot_mapping, block_size=16)
    assert passthrough is slot_mapping


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
