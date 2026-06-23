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
 * \file sparse_flash_mla_scfa_block_vector.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_SCFA_BLOCK_VECTOR_H
#define SPARSE_FLASH_MLA_SCFA_BLOCK_VECTOR_H

#include "util_regbase.h"
#include "sparse_flash_mla_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#if __has_include("../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h")
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#else
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_flashupdate_new.h")
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#else
#include "../../common/arch35/vf/vf_flashupdate_new.h"
#endif

#if __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../common/buffers_policy.h"
#endif
#if __has_include("../../common/op_kernel/buffer_manager.h")
#include "../../common/op_kernel/buffer_manager.h"
#else
#include "../common/buffer_manager.h"
#endif
#if __has_include("../../common/op_kernel/buffer.h")
#include "../../common/op_kernel/buffer.h"
#else
#include "../common/buffer.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;

namespace SMLAKernel {
TEMPLATES_DEF
class SCFABlockVec {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr float R0 = 1.0f;
    static constexpr uint32_t S2_REAL_BUF_LEN = 128;

    // ==================== Functions ======================
    __aicore__ inline SCFABlockVec() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ,
        __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
            }
            if (cuSeqlensCmpKv != nullptr) {
                cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            }
            if (seqUsedOriKV != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedOriKV);
            }
            if (seqUsedCmpKV != nullptr) {
                actualSeqLengthsCmpKVGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedCmpKV);
            }
            if (cmpResidualKV != nullptr) {
                cmpResidualKVGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKV);
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
        __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
        __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
        __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
        RunInfo &runInfo,
        ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

private:
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyToOutUb(LocalTensor<Q_T> kvNzUb, LocalTensor<KV_T> srcTensor, int64_t dealRow,
        ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        LocalTensor<Q_T> kvNzOutUb, int64_t dealRow, int64_t s2StartIdx,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo,
        ConstInfo &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb,
        int64_t startRow, int64_t keyOffset, ConstInfo &constInfo);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &constInfo);

    TPipe *tPipe;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> oriSparseIndicesGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKVGm;
    GlobalTensor<int32_t> cmpResidualKVGm;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TQue<QuePosition::VECIN, 2> stage0InQue; // for v0 input, 2份表示可能存在pingpong
    TQue<QuePosition::VECOUT, 1> stage0OutQue; // for v0 output, 2份表示可能存在pingpong
    TBuf<> stage0OutBuf[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TEventID mte3ToVAttnOutId; // 存放MTE3_V的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID vToMte3AttnOutId; // 存放V_MTE3的eventId, 用于V2 attentionOut拷出阶段的同步
 	TEventID mte3ToVLseOutId; // 存放MTE3_V的eventId, 用于V1 LSE拷出阶段的同步
 	TEventID vToMte3LseOutId; // 存放V_MTE3的eventId, 用于V1 LSE拷出阶段的同步
    TEventID mte2ToMte3[2];
    TEventID mte3ToMte2[2];
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> outLseBuf[2];

    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
    int64_t s2IdxInBase, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t sparseBlockCount = 0;
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        sparseBlockCount = constInfo.cmpSparseBlockCount;
        cmpS2LoopCnt -= runInfo.oriKvLoopEndIdx;
    }
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * sparseBlockCount +
            runInfo.s1oIdx * sparseBlockCount; // B, S1, N2(1), K
    }
    int64_t curSparseBlockCount = runInfo.isCmp ? runInfo.cmpSparseBlockCount : runInfo.oriSparseBlockCount;

    int64_t topkKIdx = s2IdxInBase + cmpS2LoopCnt * constInfo.s2BaseSize;
    if (unlikely(topkKIdx >= curSparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx + runInfo.s2StartIdx);
    }
    topkKIdx += 1;
    if (unlikely((topkKIdx >= curSparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        token1Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx + runInfo.s2StartIdx);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SCFABlockVec<TEMPLATE_ARGS>::GetkeyOffset(
    int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (IS_PA) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) *
            static_cast<int64_t>(blockSize) * constInfo.dSizeVInput +
            blkTableOffset * constInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
    } else {
        if (LAYOUT_T == SMLA_LAYOUT::BSND) {
            if (runInfo.isCmp) {
                realkeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSize + \
                    runInfo.n2oIdx * constInfo.cmpS2Size * constInfo.dSize + s2Idx * constInfo.dSize; // BSN(1)D
            } else {
                realkeyOffset = runInfo.boIdx * constInfo.n2S2Dv +
                    runInfo.n2oIdx * constInfo.s2Dv + s2Idx * constInfo.dSize; // BSN(1)D
            }
        } else if (LAYOUT_T == SMLA_LAYOUT::TND) {
            realkeyOffset = (cuSeqlensKvGm.GetValue(runInfo.boIdx) + s2Idx) *
                constInfo.n2Size * constInfo.dSize +
                runInfo.n2oIdx * constInfo.dSize; // TN(1)D
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
SCFABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset, ConstInfo &constInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = constInfo.dSize * sizeof(KV_T);

    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = (CeilAlign(constInfo.dSize * sizeof(KV_T), BUFFER_SIZE_BYTE_32B) -
 	                          constInfo.dSize * sizeof(KV_T)) / sizeof(KV_T);
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * constInfo.dSize], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t SCFABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
    int64_t token0Idx, int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    int64_t combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
    int64_t keySrcStride = (keyOffset0 > keyOffset1 ? \
        (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) * sizeof(KV_T) - combineBytes;
    if (unlikely(keyOffset1 < 0)) {
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
    } else if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) ||
        constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, constInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = combineBytes;
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) - combineBytes) / sizeof(KV_T);
        padParams.paddingValue = 0;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }
        DataCopyPad(kvInUb[startRow * constInfo.dSize], keyGm[keyOffset], intriParams, padParams);
    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyToOutUb(LocalTensor<Q_T> kvOutUb,
    LocalTensor<KV_T> srcTensor, int64_t dealRow, ConstInfo &constInfo)
{
    LocalTensor<Q_T> kvNdUb = srcTensor.template ReinterpretCast<Q_T>();
    DataCopy(kvOutUb, kvNdUb, dealRow * constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    LocalTensor<Q_T> kvNzOutUb, int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * 16], kvNzOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> kvOutUb,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * constInfo.dSize], kvOutUb, dealRow * constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        int32_t vecCnt = (aicIdx % 2U == 0) ? (GetSubBlockIdx() == 0 ? 0 : 1) : (GetSubBlockIdx() == 0 ? 2 : 3);
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = 0;
            } else {
                sparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2);
            }
        }
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        sparseCalSize = GetSubBlockIdx() == 0 ? v0S2SizeFirstCore : runInfo.s2RealSize - v0S2SizeFirstCore;
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : v0S2SizeFirstCore;
        sparseS2End = sparseS2Start + sparseCalSize;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) {
            if constexpr (IS_SPLIT_G) {
                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
            }
            return;
        }
        keyGm = cmpKVGm;
        cuSeqlensKvGm = cuSeqlensCmpKvGm;
        sparseIndicesGm = cmpSparseIndicesGm;
        if constexpr (IS_PA) {
            blockTableGm = cmpBlockTableGm;
            blockSize = constInfo.cmpBlockSize;
            maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        }
        CalSparseCalSize(runInfo, constInfo);
        ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo, startPos);
        if constexpr (IS_SPLIT_G) {
            CrossCoreSetFlag<0, PIPE_MTE3>(15);
            CrossCoreWaitFlag<0, PIPE_MTE3>(15);
        }
        outputL1.SetCrossCore();
        v0ResGm.SetCrossCore();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        if (sparseCalSize == 0) {
            return;
        }
        bool meetEnd = false;
        int64_t s2Start = sparseS2Start;
        int64_t s2 = sparseS2Start;
        int64_t token0Idx;
        int64_t token1Idx;
        uint32_t pingPong = 0;
        while ((s2 < sparseS2End) && !meetEnd) {
            int64_t dealRow = 0;
            LocalTensor<Q_T> stage0OutUb = this->stage0OutBuf[pingPong].template Get<Q_T>();
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            while (dealRow < Min(16, sparseCalSize) && s2 < sparseS2End) {
                GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
                s2 += 2;
                if (token0Idx == -1 && token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
                dealRow += CopyInKvSparse(stage0OutUb, dealRow, token0Idx, token1Idx, runInfo, constInfo);
                if (token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
            }
            if (dealRow == 0) {
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
                pingPong ^= 1;
                return;
            }
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
            CopyOutKvUb2Gm(v0ResGm, stage0OutUb, dealRow, s2Start, runInfo, constInfo);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            s2Start += dealRow;
            pingPong ^= 1;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
    RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    runInfo.s2RealSizeUpdate = runInfo.s2RealSize;

    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (runInfo.s2LoopCount == 0 && !isSinks) {
        if (likely(runInfo.s2RealSize == 128 && runInfo.s2RealSizeUpdate == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128 || runInfo.s2RealSizeUpdate < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (runInfo.s2LoopCount == 0 && isSinks) {
            // s1切1,vec0: 0 ~ halfMRealSize - 1, vec1: gSize - halfMRealSize ~ gSize
            int64_t sinksOffset = 0;
            if constexpr (!IS_SPLIT_G) {
                sinksOffset = GetBlockIdx() % 2 == 0 ? 0 : constInfo.gSize - runInfo.halfMRealSize;
            } else {
                switch (constInfo.aivIdx % 4) {
                    case 0:
                        sinksOffset = 0;
                        break;
                    case 1:
                        sinksOffset = 32;
                        break;
                    case 2:
                        sinksOffset = 64;
                        break;
                    case 3:
                        sinksOffset = 96;
                        break;
                    default:
                        break;
                }
            }
            LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
            DataCopy(maxUb, sinksUb[sinksOffset], runInfo.halfMRealSize);
            DuplicateSumWithR0<T>(sumUb, R0, runInfo.halfMRealSize);
        }
        if (likely(runInfo.s2RealSize == 128 && runInfo.s2RealSizeUpdate == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128 || runInfo.s2RealSizeUpdate < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx *
            (BLOCK_BYTE / sizeof(Q_T)) * (runInfo.mRealSize - runInfo.halfMRealSize)],
            stage1CastTensor, {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize,
            (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
            (uint16_t)(s1BaseSize - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0 || (runInfo.s2LoopCount == 0 && isSinks)) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if (constInfo.returnSoftmaxLse && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        LocalTensor<float> outLse = this->outLseBuf[runInfo.multiCoreIdxMod2].template Get<float>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(float) * runInfo.halfMRealSize;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        ComputeLseOutputVF<float>(outLse, sumUb, maxUb, runInfo.halfMRealSize);
        SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
 	    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
        DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], outLse, dataCopyParams);
        SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        if (runInfo.s2RealSizeUpdate > 0) {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb,
                    runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            }
        } else {
            if (runInfo.s2LoopCount >= runInfo.s2LoopLimit) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut (RunInfo &runInfo, ConstInfo &constInfo,
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo &runInfo,
    ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUTPUT_T>(
                this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CleanOutput(
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxLseGm.SetGlobalBuffer((__gm__ T *)softmaxLse);
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
    __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
        }
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo &constInfo)
{
    LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
    TEventID mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    // ub buffer
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512); // sinksBuf内存申请512B
    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        tPipe->InitBuffer(stage0OutBuf[0], dVTemplateType * 16 * sizeof(KV_T));
        tPipe->InitBuffer(stage0OutBuf[1], dVTemplateType * 16 * sizeof(KV_T));
    }
    if (constInfo.returnSoftmaxLse) {
        tPipe->InitBuffer(outLseBuf[0], 256);
        tPipe->InitBuffer(outLseBuf[1], 256);
    }

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVAttnOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    vToMte3AttnOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    mte3ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    mte3ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
    mte2ToMte3[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToMte3[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF
class SCFABlockVecDummy {
public:
    __aicore__ inline SCFABlockVecDummy() {};
    __aicore__ inline void CleanOutput(
        __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
        __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
        __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
        __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ,
        __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
};
}
#endif // SPARSE_FLASH_MLA_SCFA_BLOCK_VECTOR_H

