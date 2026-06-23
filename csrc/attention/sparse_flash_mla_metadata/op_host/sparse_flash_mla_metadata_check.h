/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_mla_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../sparse_flash_mla/op_kernel/sparse_flash_mla_metadata.h"
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

namespace {

enum class SparseModeSmla : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

inline constexpr int64_t SMLA_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t SMLA_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t SMLA_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t SMLA_NUM_HEADS_Q_UPPER_BOUND = 128;

inline bool IsPowerOfTwoInRangeSmla(int64_t value, int64_t minValue, int64_t maxValue)
{
    return value >= minValue && value <= maxValue && ((value & (value - 1)) == 0);
}

inline bool IsA5Smla(const char *socVersion)
{
    return socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr;
}

inline bool IsTensorExistSmla(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

aclnnStatus CheckReservedOptionalTensorSmla(const aclTensor *tensor, const char *tensorName)
{
    if (!IsTensorExistSmla(tensor)) {
        return ACLNN_SUCCESS;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "%s is reserved and does not support non-empty tensor in current version", tensorName);
    return ACLNN_ERR_PARAM_INVALID;
}

int64_t GetDimNumSmla(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeSmla(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

aclnnStatus CheckSingleParamSmla(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv, int64_t maxSeqlenCmpKv,
                                 int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t oriTopk,
                                 int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
                                 int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
                                 const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, uint32_t aicCoreNum,
                                 uint32_t aivCoreNum, const char *socVersion)
{
    // batch_size >= 0
    if (batchSize < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should be >= 0, but got %lld", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q >= 0
    if (maxSeqlenQ < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should be >= 0, but got %lld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_ori_kv >= 0
    if (maxSeqlenOriKv < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_ori_kv should be >= 0, but got %lld", maxSeqlenOriKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_cmp_kv >= 0
    if (maxSeqlenCmpKv < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_cmp_kv should be >= 0, but got %lld", maxSeqlenCmpKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q [1, 128]
    if (numHeadsQ < SMLA_NUM_HEADS_Q_LOWER_BOUND || numHeadsQ > SMLA_NUM_HEADS_Q_UPPER_BOUND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q should be [%lld, %lld], but got %lld",
            SMLA_NUM_HEADS_Q_LOWER_BOUND, SMLA_NUM_HEADS_Q_UPPER_BOUND, numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv: 1
    if (numHeadsKv != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %lld", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsQ % numHeadsKv != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "num_heads_q should be divisible by num_heads_kv, but got %lld and %lld", numHeadsQ, numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t headRatio = numHeadsQ / numHeadsKv;
    bool isA5 = IsA5Smla(socVersion);
    if (isA5) {
        if (headRatio < SMLA_NUM_HEADS_Q_LOWER_BOUND || headRatio > SMLA_NUM_HEADS_Q_UPPER_BOUND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q / num_heads_kv should be in [%lld, %lld], but got %lld",
                SMLA_NUM_HEADS_Q_LOWER_BOUND, SMLA_NUM_HEADS_Q_UPPER_BOUND, headRatio);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (!IsPowerOfTwoInRangeSmla(
        headRatio, SMLA_NUM_HEADS_Q_LOWER_BOUND, SMLA_NUM_HEADS_Q_UPPER_BOUND)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "num_heads_q / num_heads_kv should be power of two in [%lld, %lld], but got %lld",
                SMLA_NUM_HEADS_Q_LOWER_BOUND, SMLA_NUM_HEADS_Q_UPPER_BOUND, headRatio);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim: 512
    if (headDim != 512) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_dim should only be 512, but got %lld", headDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hasOriKv) {
        // ori_topk >= 0
        if (oriTopk < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_topk should be >= 0, but got %lld", oriTopk);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!isA5 && oriTopk != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ori_topk is reserved and should only be 0, but got %lld", oriTopk);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_mask_mode: 0, 3, or 4
        if (oriMaskMode != static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK) &&
            oriMaskMode != static_cast<int64_t>(SparseModeSmla::RIGHT_DOWN_CAUSAL) &&
            oriMaskMode != static_cast<int64_t>(SparseModeSmla::BAND)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_mask_mode should be 0, 3 or 4, but got %lld",
                oriMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_win_left >= -1
        if (oriWinLeft < -1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_win_left should be >= -1, but got %lld",
                oriWinLeft);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_win_right >= -1
        if (oriWinRight < -1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_win_right should be >= -1, but got %lld",
                oriWinRight);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasCmpKv) {
        // cmp_topk >= 0
        if (cmpTopk < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_topk should be >= 0, but got %lld", cmpTopk);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!isA5 && cmpTopk != 0 && cmpTopk != 512 && cmpTopk != 1024) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_topk should be 0, 512 or 1024, but got %lld", cmpTopk);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_mask_mode: 0 or 3
        if (cmpMaskMode != static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK) &&
            cmpMaskMode != static_cast<int64_t>(SparseModeSmla::RIGHT_DOWN_CAUSAL)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_mask_mode should be 0 or 3, but got %lld",
                cmpMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (isA5) {
            if (cmpRatio < SMLA_CMP_RATIO_LOWER_BOUND || cmpRatio > SMLA_CMP_RATIO_UPPER_BOUND) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be in [%lld, %lld], but got %lld",
                    SMLA_CMP_RATIO_LOWER_BOUND, SMLA_CMP_RATIO_UPPER_BOUND, cmpRatio);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else if (cmpRatio != 1 && cmpRatio != 4 && cmpRatio != 128) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be 1, 4 or 128, but got %lld", cmpRatio);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (layoutQOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q is null!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutKvOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv is null!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q: BSND or TND
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv: BSND, TND, or PA_BBND
    if ((strcmp(layoutKvOptional, "BSND") != 0) && (strcmp(layoutKvOptional, "TND") != 0) &&
        (strcmp(layoutKvOptional, "PA_BBND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_BBND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核数校验
    if (aicCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %u", aicCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aicCoreNum > optiling::AIC_CORE_MAX_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The maximum supported AIC num is %u, but got %u", optiling::AIC_CORE_MAX_NUM,
            aicCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIV num should be larger than 0, but got %u", aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum > optiling::AIV_CORE_MAX_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The maximum supported AIV num is %u, but got %u", optiling::AIV_CORE_MAX_NUM,
            aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验切g模板核数
    if (numHeadsQ == 128) {
        if (aicCoreNum == 1 || aivCoreNum == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When num_heads_q is 128, AIC num and AIV num should not be 1, "
                "but got %u and %u", aicCoreNum, aivCoreNum);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSmla(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                               const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedOriKvOptional,
                               const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
                               const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional,
                               int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
                               int64_t cmpMaskMode, bool hasOriKv, bool hasCmpKv, const char *layoutQOptional,
                               const char *layoutKvOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (!IsTensorExistSmla(cuSeqlensQOptional)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is TND, cu_seqlens_q must be provided!");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasOriKv) {
        // cu_seqlens_ori_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            if (!IsTensorExistSmla(cuSeqlensOriKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true and layout_kv is TND, "
                    "cu_seqlens_ori_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // seqused_ori_kv 存在性校验
        if (strcmp(layoutKvOptional, "PA_BBND") == 0) {
            if (!IsTensorExistSmla(sequsedOriKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true and layout_kv is PA_BBND, "
                    "seqused_ori_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // ori_topk_length 存在性校验
        if (oriTopk != 0 && oriMaskMode == static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK)) {
            if (!IsTensorExistSmla(oriTopkLengthOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_topk is not 0 and ori_mask_mode is 0, "
                    "ori_topk_length must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    if (hasCmpKv) {
        // cu_seqlens_cmp_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            if (!IsTensorExistSmla(cuSeqlensCmpKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true and layout_kv is TND, "
                    "cu_seqlens_cmp_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // seqused_cmp_kv 存在性校验
        if (strcmp(layoutKvOptional, "PA_BBND") == 0) {
            if (!IsTensorExistSmla(sequsedCmpKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true and layout_kv is PA_BBND, "
                    "seqused_cmp_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // cmp_residual_kv 存在性校验
        if (cmpRatio != 1 && cmpMaskMode == static_cast<int64_t>(SparseModeSmla::RIGHT_DOWN_CAUSAL)) {
            if (!IsTensorExistSmla(cmpResidualKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_ratio is not 1 and cmp_mask_mode is 3, "
                    "cmp_residual_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // cmp_topk_length 存在性校验
        if (cmpTopk != 0 && cmpMaskMode == static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK)) {
            if (!IsTensorExistSmla(cmpTopkLengthOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_topk is not 0 and cmp_mask_mode is 0, "
                    "cmp_topk_length must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // metadata 存在性校验
    if (!IsTensorExistSmla(metadata)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output metadata is nullptr!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeSmla(const aclTensor *sequsedQOptional, const aclTensor *cuSeqlensQOptional,
                              const char *layoutQOptional, int64_t batchSize)
{
    // 1. 如果 sequsedQOptional 传了，使用 sequsedQOptional 获取 BatchSize
    if (IsTensorExistSmla(sequsedQOptional)) {
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedQOptional 没传，判断 Layout
    if (strcmp(layoutQOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqLensQOptional 获取 BatchSize
        if (IsTensorExistSmla(cuSeqlensQOptional)) {
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqLensQOptional 为空，使用 batchSize
    return batchSize;
}

int64_t GetOriKvBatchSizeSmla(const aclTensor *sequsedOriKvOptional, const aclTensor *cuSeqlensOriKvOptional,
                              const char *layoutKvOptional, int64_t batchSize)
{
    // 1. 如果 sequsedOriKvOptional 传了，使用 sequsedOriKvOptional 获取 BatchSize
    if (IsTensorExistSmla(sequsedOriKvOptional)) {
        return sequsedOriKvOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedOriKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqlensOriKvOptional 获取 BatchSize
        if (IsTensorExistSmla(cuSeqlensOriKvOptional)) {
            return cuSeqlensOriKvOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensOriKvOptional 为空，使用 batchSize
    return batchSize;
}

int64_t GetCmpKvBatchSizeSmla(const aclTensor *sequsedCmpKvOptional, const aclTensor *cuSeqlensCmpKvOptional,
                              const char *layoutKvOptional, int64_t batchSize)
{
    // 1. 如果 sequsedCmpKvOptional 传了，使用 sequsedCmpKvOptional 获取 BatchSize
    if (IsTensorExistSmla(sequsedCmpKvOptional)) {
        return sequsedCmpKvOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedCmpKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqlensCmpKvOptional 获取 BatchSize
        if (IsTensorExistSmla(cuSeqlensCmpKvOptional)) {
            return cuSeqlensCmpKvOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensCmpKvOptional 为空，使用 batchSize
    return batchSize;
}

aclnnStatus CheckConsistencySmla(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                                 const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                                 const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                                 const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                                 const aclTensor *cmpTopkLengthOptional, int64_t batchSize, const char *layoutQOptional,
                                 const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, bool isA5,
                                 const aclTensor *metadata)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    int64_t dimNum = -1;
    if (!isA5 &&
        (CheckReservedOptionalTensorSmla(oriTopkLengthOptional, "ori_topk_length") != ACLNN_SUCCESS ||
         CheckReservedOptionalTensorSmla(cmpTopkLengthOptional, "cmp_topk_length") != ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 cu_seqlens_q
    if (IsTensorExistSmla(cuSeqlensQOptional)) {
        // 校验 cu_seqlens_q 维度
        dimNum = GetDimNumSmla(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %lld", dimNum);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_q 数据类型
        dataType = GetDataTypeSmla(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistSmla(sequsedQOptional)) {
        // 校验 seqused_q 维度
        dimNum = GetDimNumSmla(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %lld", dimNum);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_q 数据类型
        dataType = GetDataTypeSmla(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // ori_kv部分
    if (hasOriKv) {
        // 校验 cu_seqlens_ori_kv
        if (IsTensorExistSmla(cuSeqlensOriKvOptional)) {
            // 校验 cu_seqlens_ori_kv 维度
            dimNum = GetDimNumSmla(cuSeqlensOriKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_ori_kv must be 1, but got %lld",
                    dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_ori_kv 数据类型
            dataType = GetDataTypeSmla(cuSeqlensOriKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_ori_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_ori_kv
        if (IsTensorExistSmla(sequsedOriKvOptional)) {
            // 校验 seqused_ori_kv 维度
            dimNum = GetDimNumSmla(sequsedOriKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_ori_kv must be 1, but got %lld",
                    dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_ori_kv 数据类型
            dataType = GetDataTypeSmla(sequsedOriKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_ori_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 ori_topk_length
        if (IsTensorExistSmla(oriTopkLengthOptional)) {
            // 校验 ori_topk_length 维度
            dimNum = GetDimNumSmla(oriTopkLengthOptional);
            if (strcmp(layoutQOptional, "TND") == 0) {
                if (dimNum != 2) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is TND, the dim num of ori_topk_length must be 2, "
                            "but got %lld", dimNum);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else if (strcmp(layoutQOptional, "BSND") == 0) {
                if (dimNum != 3) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is BSND, the dim num of ori_topk_length must be 3, "
                            "but got %lld", dimNum);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 ori_topk_length 数据类型
            dataType = GetDataTypeSmla(oriTopkLengthOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of ori_topk_length must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // cmp_kv部分
    if (hasCmpKv) {
        // 校验 cu_seqlens_cmp_kv
        if (IsTensorExistSmla(cuSeqlensCmpKvOptional)) {
            // 校验 cu_seqlens_cmp_kv 维度
            dimNum = GetDimNumSmla(cuSeqlensCmpKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_cmp_kv must be 1, but got %lld",
                    dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_cmp_kv 数据类型
            dataType = GetDataTypeSmla(cuSeqlensCmpKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_cmp_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_cmp_kv
        if (IsTensorExistSmla(sequsedCmpKvOptional)) {
            // 校验 seqused_cmp_kv 维度
            dimNum = GetDimNumSmla(sequsedCmpKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_cmp_kv must be 1, but got %lld",
                    dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_cmp_kv 数据类型
            dataType = GetDataTypeSmla(sequsedCmpKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_cmp_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_residual_kv
        if (IsTensorExistSmla(cmpResidualKvOptional)) {
            // 校验 cmp_residual_kv 维度
            dimNum = GetDimNumSmla(cmpResidualKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cmp_residual_kv must be 1, but got %lld",
                    dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cmp_residual_kv 数据类型
            dataType = GetDataTypeSmla(cmpResidualKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cmp_residual_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_topk_length
        if (IsTensorExistSmla(cmpTopkLengthOptional)) {
            // 校验 cmp_topk_length 维度
            dimNum = GetDimNumSmla(cmpTopkLengthOptional);
            if (strcmp(layoutQOptional, "TND") == 0) {
                if (dimNum != 2) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is TND, the dim num of cmp_topk_length must be 2, "
                            "but got %lld", dimNum);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else if (strcmp(layoutQOptional, "BSND") == 0) {
                if (dimNum != 3) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is BSND, the dim num of cmp_topk_length must be 3, "
                            "but got %lld", dimNum);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 cmp_topk_length 数据类型
            dataType = GetDataTypeSmla(cmpTopkLengthOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cmp_topk_length must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // 校验 metadata
    if (IsTensorExistSmla(metadata)) {
        // 校验 metadata 维度
        dimNum = GetDimNumSmla(metadata);
        if (dimNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lld", dimNum);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != optiling::SMLA_METADATA_TOTAL_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %lld",
                optiling::SMLA_METADATA_TOTAL_SIZE, metadata->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 数据类型
        dataType = GetDataTypeSmla(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of metadata must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 q/kv 维度一致性
    int64_t queryBatchSize = GetQueryBatchSizeSmla(sequsedQOptional, cuSeqlensQOptional, layoutQOptional, batchSize);
    if (hasOriKv) {
        int64_t oriKvBatchSize = GetOriKvBatchSizeSmla(
            sequsedOriKvOptional, cuSeqlensOriKvOptional, layoutKvOptional, batchSize);
        if (queryBatchSize != oriKvBatchSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when has_ori_kv is true, the batch_size obtained from q should be "
                    "the same as that obtained from ori_kv, but got %lld and %lld", queryBatchSize,
                    oriKvBatchSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasCmpKv) {
        int64_t cmpKvBatchSize = GetCmpKvBatchSizeSmla(
            sequsedCmpKvOptional, cuSeqlensCmpKvOptional, layoutKvOptional, batchSize);
        if (queryBatchSize != cmpKvBatchSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when has_cmp_kv is true, the batch_size obtained from q should be "
                    "the same as that obtained from cmp_kv, but got %lld and %lld", queryBatchSize,
                    cmpKvBatchSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cmp_residual_kv 元素数
        if (IsTensorExistSmla(cmpResidualKvOptional)) {
            if (cmpResidualKvOptional->GetViewShape().GetDim(0) != queryBatchSize) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The elements num of cmp_residual_kv should match the valid batch "
                    "size, but got %lld and %lld", cmpResidualKvOptional->GetViewShape().GetDim(0), queryBatchSize);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                               const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                               const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                               const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                               const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ, int64_t numHeadsKv,
                               int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
                               int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio,
                               int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
                               const char *layoutQOptional, const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
                               uint32_t aicCoreNum, uint32_t aivCoreNum, const char *socVersion,
                               const aclTensor *metaData)
{
    bool isA5 = IsA5Smla(socVersion);
    if (CheckSingleParamSmla(batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, numHeadsQ, numHeadsKv, headDim,
                             oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                             layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, aicCoreNum, aivCoreNum,
                             socVersion) == ACLNN_SUCCESS &&
        CheckExistenceSmla(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedOriKvOptional,
                           sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional,
                           oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, hasOriKv, hasCmpKv, layoutQOptional,
                           layoutKvOptional, metaData) == ACLNN_SUCCESS &&
        CheckConsistencySmla(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                             sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                             cmpTopkLengthOptional, batchSize, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv,
                             isA5, metaData) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}
}

#ifdef __cplusplus
}
#endif
