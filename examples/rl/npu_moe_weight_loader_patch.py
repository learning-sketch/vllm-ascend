# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM worker extension that keeps MoE params loadable across weight updates.

Why this is needed
------------------
For MoE models, ``AscendUnquantizedFusedMoEMethod.process_weights_after_loading``
(and the quantized variants) re-create the fused expert parameters
(``w13_weight`` / ``w2_weight``) as brand-new ``torch.nn.Parameter`` objects
after the first load. The new parameters lose the extra attributes vLLM attached
during loading -- most importantly ``weight_loader``. On a SECOND load (an RLHF
weight update / refit), ``model.load_weights`` does ``param.weight_loader`` and
fails with::

    AttributeError: 'Parameter' object has no attribute 'weight_loader'

This is a known vLLM MoE issue (vllm-project/vllm#17429, #17915, #16842; fixed on
GPU by #16854). The vLLM RL frameworks work around it on the worker side: verl's
``patch_vllm_moe_model_weight_loader`` and vime's NPU PR (vllm-project/vime#269)
re-attach the loader via ``--worker-extension-cls``. This module is the
vllm-ascend equivalent.

What it does
------------
At import time (which happens inside every worker process when vLLM resolves
``--worker-extension-cls``, before the model is loaded) it wraps the Ascend MoE
``process_weights_after_loading`` so that any extra attributes present on the
expert parameters before processing -- including ``weight_loader`` -- are
restored onto the (possibly re-created) parameters afterwards.

Usage
-----
Put this file on PYTHONPATH and pass its class to ``vllm serve``::

    PYTHONPATH=/path/to/examples/rl:$PYTHONPATH \
    vllm serve <model> \
        --weight-transfer-config '{"backend": "ipc"}' \
        --tensor-parallel-size 4 --enable-expert-parallel \
        --worker-extension-cls npu_moe_weight_loader_patch.MoEWeightLoaderWorkerExtension \
        ...

With this patch the ``weight_loader`` error no longer occurs, so the MoE weight
update succeeds. (``--enforce-eager`` may still be advisable for other aclgraph +
layerwise-reload interactions, but it is no longer required to avoid this error.)
"""

import functools

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001 - logging is best-effort
    import logging

    logger = logging.getLogger(__name__)


_PATCH_FLAG = "_moe_weight_loader_preserved"


def _wrap_process_weights_after_loading(cls: type) -> bool:
    """Wrap ``cls.process_weights_after_loading`` to preserve param extra attrs.

    Idempotent: a class is only wrapped once. Returns True if it wrapped now.
    """
    if cls is None or getattr(cls, _PATCH_FLAG, False):
        return False
    original = cls.process_weights_after_loading

    @functools.wraps(original)
    def wrapped(self, layer, *args, **kwargs):
        # Snapshot extra attributes (weight_loader, expert_id maps, output_dim,
        # ...) of the layer's direct parameters before processing re-creates them.
        saved: dict[str, dict] = {}
        for name, param in layer.named_parameters(recurse=False):
            saved[name] = dict(vars(param))

        result = original(self, layer, *args, **kwargs)

        # Restore any attribute that the new (re-created) parameter lost.
        for name, param in layer.named_parameters(recurse=False):
            for key, value in saved.get(name, {}).items():
                if not hasattr(param, key):
                    setattr(param, key, value)
        return result

    cls.process_weights_after_loading = wrapped
    setattr(cls, _PATCH_FLAG, True)
    return True


def apply_moe_weight_loader_patch() -> None:
    """Patch the Ascend FusedMoE weight-processing methods (idempotent)."""
    patched: list[str] = []

    # Unquantized MoE (bf16/fp16) -- the common RL case.
    try:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

        if _wrap_process_weights_after_loading(AscendUnquantizedFusedMoEMethod):
            patched.append("AscendUnquantizedFusedMoEMethod")
    except Exception as exc:  # noqa: BLE001
        logger.warning("MoE weight_loader patch: skip unquantized method (%s)", exc)

    # Quantized MoE methods (best-effort; names vary across versions/quant types).
    try:
        from vllm_ascend.quantization import method_adapters

        for attr in dir(method_adapters):
            obj = getattr(method_adapters, attr)
            if (
                isinstance(obj, type)
                and "FusedMoE" in attr
                and hasattr(obj, "process_weights_after_loading")
            ):
                if _wrap_process_weights_after_loading(obj):
                    patched.append(attr)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MoE weight_loader patch: skip quantized methods (%s)", exc)

    if patched:
        logger.info("MoE weight_loader patch applied to: %s", ", ".join(patched))
    else:
        logger.warning(
            "MoE weight_loader patch found nothing to patch; "
            "check vllm-ascend version / class names."
        )


# Apply on import. ``--worker-extension-cls`` imports this module inside every
# worker process before the model is built, so the wrap is in place for the very
# first process_weights_after_loading call.
apply_moe_weight_loader_patch()


class MoEWeightLoaderWorkerExtension:
    """vLLM worker extension entry point.

    The actual fix is applied at module import (``apply_moe_weight_loader_patch``);
    this class only needs to exist so ``--worker-extension-cls`` can reference it.
    ``reapply_moe_weight_loader_patch`` is exposed so it can be re-run via
    ``collective_rpc`` if ever needed.
    """

    def reapply_moe_weight_loader_patch(self) -> bool:
        apply_moe_weight_loader_patch()
        return True
