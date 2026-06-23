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
 * \file sparse_flash_mla_scfa_block_cube.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_SCFA_BLOCK_CUBE_H_
#define SPARSE_FLASH_MLA_SCFA_BLOCK_CUBE_H_
#include "kernel_operator_list_tensor_intf.h"
#include "util_regbase.h"
#include "sparse_flash_mla_common_arch35.h"

#if __has_include("../../common/op_kernel/offset_calculator.h")
#include "../../common/op_kernel/offset_calculator.h"
#else
#include "../common/offset_calculator.h"
#endif
#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace SMLAKernel {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

template <SMLA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == SMLA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

template <SMLA_LAYOUT KV_LAYOUT_T>
__aicore__ inline constexpr GmFormat GetKvGmFormat()
{
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        return GmFormat::TND;
    } else { // BSND
        return GmFormat::BSND;
    }
}

TEMPLATES_DEF
class SCFABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;

    __aicore__ inline SCFABlockCube() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe,
        BufferManager<BufferType::L1> &l1BufferManager, __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ,
        __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
        __gm__ uint8_t *cuSeqlensCmpKv,
        __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, const ConstInfo& constInfo);

    __aicore__ inline void MergeKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, const ConstInfo &constInfo, int32_t startPos);
    // SWA/CFA场景, inputRightBuf是INNER_CORE_SYNC类型
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf,
        const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf, const RunInfo &runInfo,
        const ConstInfo &constInfo);
    // SCFA场景, inputRightBuf是CROSS_CORE_SYNC_FORWARD类型
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
        const ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer(BufferManager<BufferType::L1> &l1BufferManager);
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
        __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
        __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, const ConstInfo& constInfo);
    __aicore__ inline void CalcS1Coord(const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void CalcS2Coord(const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, const ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<Q_T> inputRightTensor, int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1CFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf,
        const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2CFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf, const RunInfo &runInfo,
        const ConstInfo &constInfo);

    __aicore__ inline void IterateBmm1SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
        const ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKvGmFormat<KV_LAYOUT_T>();
    static constexpr bool Q_WITH_ZERO_HEAD = (LAYOUT_T == SMLA_LAYOUT::TND);
    FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_WITH_ZERO_HEAD> queryGm;
    static constexpr bool KV_WITH_ZERO_HEAD = (KV_LAYOUT_T == SMLA_LAYOUT::TND);
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> oriKvGm;
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> cmpKvGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> curKvGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    TEventID mte1ToMte2Id[3];
    TEventID mte2ToMte1Id[3];

    /* =====================LocalBuffer变量==================== */
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    // D小于等于256 mm1左矩阵Q，GS1循环内左矩阵复用, GS1循环间开pingpong；D大于256使用单块Buffer，S1循环间驻留；fp32场景单块不驻留
    BuffersPolicySingleBuffer<BufferType::L1> l1QBuffers;

    // L0A
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    // L0B
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    // L0C
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, BufferManager<BufferType::L1> &l1BufferManager, __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer(l1BufferManager);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, const ConstInfo& constInfo)
{
    if ASCEND_IS_AIC {
        this->oriKvGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)oriKv);
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            this->oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
        }
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            this->cmpKvGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)cmpKv);
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                this->cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
            }
        }
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
            this->cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
        }
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE || \
                      IS_SPLIT_G) {
            mte1ToMte2Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte1ToMte2Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte1ToMte2Id[2] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte2ToMte1Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            mte2ToMte1Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            mte2ToMte1Id[2] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        }
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
            SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[0]);
            SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[1]);
            SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[2]);
        }
        InitGmTensor(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitLocalBuffer(BufferManager<BufferType::L1> &l1BufferManager)
{
    constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(Q_T);
    l1QBuffers.Init((l1BufferManager), mm1LeftSize);

    // L0A B C 当前写死，能否通过基础api获取
    l0aBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0bBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0cBufferManager.Init(tPipe, L0C_SHARED_SIZE_256K);

    mmL0ABuffers.Init(l0aBufferManager, BUFFER_SIZE_16K); // db类型，填入数值是总大小的一半
    mmL0BBuffers.Init(l0bBufferManager, BUFFER_SIZE_32K);
    mmL0CBuffers.Init(l0cBufferManager, BUFFER_SIZE_128K);
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, const ConstInfo& constInfo)
{
    if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSize);
    } else {  // SMLA_LAYOUT::TND
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        GlobalTensor<int32_t> sequsedQGm;
        if (sequsedQ != nullptr) {
            sequsedQGm.SetGlobalBuffer((__gm__ int32_t *)sequsedQ);
        }
        uint32_t sequsedQSize = (sequsedQ == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensQGm, sequsedQGm, constInfo.bSize, sequsedQSize);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
            parser);
    }

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        this->oriKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.oriBlockSize, constInfo.dSize,
            this->oriBlockTableGm, constInfo.oriMaxBlockNumPerBatch);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            this->cmpKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.cmpBlockSize, constInfo.dSize,
                this->cmpBlockTableGm, constInfo.cmpMaxBlockNumPerBatch);
        }
    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
        GlobalTensor<int32_t> seqUsedOriKvGm;
        if (seqUsedOriKV != nullptr) {
            seqUsedOriKvGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedOriKV);
        }
        uint32_t seqUsedOriKvSize = (seqUsedOriKV == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensOriKvGm, seqUsedOriKvGm, constInfo.actualSeqLenSize, seqUsedOriKvSize);
        this->oriKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.dSize, parser);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            GlobalTensor<int32_t> seqUsedCmpKvGm;
            if (seqUsedCmpKV != nullptr) {
                seqUsedCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedCmpKV);
            }
            uint32_t seqUseCmpKvSize = (seqUsedCmpKV == nullptr) ? 0 : constInfo.bSize;
            ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
            parser.Init(cuSeqlensCmpKvGm, seqUsedCmpKvGm, constInfo.actualSeqLenSize, seqUseCmpKvSize);
            this->cmpKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.dSize, parser);
        }
    } else {
        // BSND不需要初始化
        this->oriKvGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size,
                constInfo.s2Size, constInfo.dSize);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            this->cmpKvGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size,
                constInfo.cmpS2Size, constInfo.dSize); // 替换constInfo.s2Size / constInfo.cmpRatio
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CalcS1Coord(const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    // 计算s1方向偏移
    coordInfo[runInfo.taskIdMod3].s1Coord = runInfo.s1oIdx * runInfo.qSNumInOneBlock;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CalcS2Coord(const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    // 计算s2方向偏移
    coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.boIdx;
    if (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) {
        kvCacheBlockSize = constInfo.cmpBlockSize;
        maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        coordInfo[runInfo.taskIdMod3].s2Coord = runInfo.s2StartIdx + \
            (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) * s2BaseSize;
        blockTableGm = cmpBlockTableGm;
        curKvGm = cmpKvGm;
    } else {
        kvCacheBlockSize = constInfo.oriBlockSize;
        maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
        coordInfo[runInfo.taskIdMod3].s2Coord = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;
        blockTableGm = oriBlockTableGm;
        curKvGm = oriKvGm;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::MergeKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, const ConstInfo &constInfo, int32_t startPos)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) { // ori kv阶段无核间操作
            return;
        }
        if (runInfo.s2RealSize > startPos) {
            CalcS2Coord(runInfo, constInfo);
            ProcessSparseKv(outputL1, runInfo, constInfo, startPos);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, const ConstInfo &constInfo, int32_t startPos)
{
    int64_t calNum = (runInfo.s2RealSize - startPos > 16) ? 16 : runInfo.s2RealSize - startPos; // 单次最多处理16行s2
    int64_t s2Loops = CeilDiv(calNum, 2); // 每次循环处理2个topk

    bool meetEnd = false;
    int64_t s2 = startPos;
    int64_t s2End = s2Loops * 2 + startPos;
    int64_t token0Idx; // 拷贝进入的两个token的index
    int64_t token1Idx;
    // 处理一个s2的base块
    int64_t dealRow = startPos;
    while ((s2 < s2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        // 1、copy kv in, gm ->ub
        LocalTensor<Q_T> kvInL1 = outputL1.GetTensor<Q_T>();
        GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
        s2 += 2; // 每次搬运2行
        if (token0Idx== -1 && token1Idx == -1) {
            meetEnd = true;
            break;
        }
        dealRow += CopyInKvSparse(kvInL1, dealRow, token0Idx, token1Idx, runInfo, constInfo);
        if (token1Idx == -1) {
            meetEnd = true;
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
    int64_t s2IdxInBase, const RunInfo &runInfo, const ConstInfo &constInfo)
{
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.cmpSparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.cmpSparseBlockCount +
            runInfo.s1oIdx * constInfo.cmpSparseBlockCount; // B, S1, N2(1), K
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx;
    int64_t topkKIdx = s2IdxInBase + cmpS2LoopCnt * constInfo.s2BaseSize;
    if (unlikely(topkKIdx >= constInfo.cmpSparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = cmpSparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
    }
    topkKIdx += 1;
    if (unlikely(topkKIdx >= constInfo.cmpSparseBlockCount)) {
        token1Idx = -1;
    } else {
        token1Idx = cmpSparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t SCFABlockCube<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<Q_T> inputRightTensor,
    int64_t startRow, int64_t token0Idx, int64_t token1Idx, const RunInfo &runInfo, const ConstInfo &constInfo)
{
    if constexpr (IS_PA) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = token0Idx;
        startPos.dIdx = 0;
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = 1;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        LocalTensor<Q_T> l1Tensor = inputRightTensor[startRow * 16];
        GmCopyInToL1PA<KV_T>(l1Tensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
        if (token1Idx >= 0) {
            startPos.s2Offset = token1Idx;
            l1Tensor = inputRightTensor[(startRow + 1) * 16];
            GmCopyInToL1PA<KV_T>(l1Tensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
        }
    }
    return (token0Idx >= 0) + (token1Idx >= 0);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    CalcS1Coord(runInfo, constInfo);
    CalcS2Coord(runInfo, constInfo);

    IterateBmm1CFA(outputBuf, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    IterateBmm2CFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1CFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    Buffer<BufferType::L1> inputLeftBuf;
    // 左矩阵复用，S2的第一次循环加载左矩阵
    // 加载左矩阵到L1, 全载
    // query对ori_kv, cmp_kv都一样，无需区分
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运Q
        inputLeftBuf = l1QBuffers.Get();
        inputLeftBuf.Wait<HardEvent::MTE1_MTE2>(); // 占用L1A
        LocalTensor<Q_T> inputLeftTensor = inputLeftBuf.GetTensor<Q_T>();

        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(
            runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx, coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<Q_T>(inputLeftTensor, this->queryGm.gmTensor[gmOffset], runInfo.mRealSize, constInfo.dSize,
            constInfo.mm1Ka);

        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非S2的第一次循环直接复用Q
        inputLeftBuf = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    }

    // 加载当前轮的右矩阵到L1
    inputRightBuf.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
    LocalTensor<KV_T> inputRightTensor = inputRightBuf.GetTensor<KV_T>();
    if constexpr (IS_PA) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = coordInfo[runInfo.taskIdMod3].s2Coord;
        startPos.dIdx = 0;
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        GmCopyInToL1PA<KV_T>(inputRightTensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
    } else {
        int64_t keyOffset = this->curKvGm.offsetCalculator.GetOffset(
            coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<KV_T>(inputRightTensor, curKvGm.gmTensor[keyOffset], runInfo.s2RealSize,
                            constInfo.dSize, constInfo.mm1Kb);
    }
    inputRightBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    inputRightBuf.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                        static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                        static_cast<uint32_t>(constInfo.dSize),   // singleK
                        0,    // isLeftTranspose
                        1     // isRightTranspose
                    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(  // m,n不切，k切128
        inputLeftBuf.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(),                // mm1B直接用tensor的数据
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        inputLeftBuf.Set<HardEvent::MTE1_MTE2>(); // 释放L1A
    }

    mm1ResL0C.Set<HardEvent::M_FIX>();    // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>();   // 等待L0C

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
    // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dstStride = s2BaseSize;
    // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    // 将matmul结果从L0C搬运到UB
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams);
    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2CFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> inputLeftBuf = inputLeftBuffers.Get(); // P直接用无需搬运
    inputLeftBuf.WaitCrossCore();

    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {static_cast<uint32_t>(s1BaseSize),          // singleM 64
                     static_cast<uint32_t>(constInfo.dSizeV), // singleN 512
                     static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(),
        inputRightBuf.GetTensor<Q_T>(),
        mmL0ABuffers,
        mmL0BBuffers,
        mm2ResL0C.GetTensor<T>(),
        param);

    inputLeftBuf.SetCrossCore();
    // bmm2才释放KV，在这里释放
    inputRightBuf.Set<HardEvent::MTE1_MTE2>();

    mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

    // 占用
    outputBuf.WaitCrossCore();
    // L0C→UB;FixpipeParamsM300:L0C→UB
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.mSize = s1BaseSize;
    // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.srcStride = Align16Func(s1BaseSize);
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(
        outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    CalcS1Coord(runInfo, constInfo);
    CalcS2Coord(runInfo, constInfo);

    IterateBmm1SCFA(outputBuf, inputRightBuf, v0ResGm, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    IterateBmm2SCFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1SCFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    Buffer<BufferType::L1> inputLeftBuf;
    // 左矩阵复用，S2的第一次循环加载左矩阵
    // 加载左矩阵到L1, 全载
    // query对ori_kv, cmp_kv都一样，无需区分
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运Q
        inputLeftBuf = l1QBuffers.Get();
        inputLeftBuf.Wait<HardEvent::MTE1_MTE2>(); // 占用L1A
        LocalTensor<Q_T> inputLeftTensor = inputLeftBuf.GetTensor<Q_T>();

        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
            coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<Q_T>(inputLeftTensor, this->queryGm.gmTensor[gmOffset], runInfo.mRealSize, constInfo.dSize,
            constInfo.mm1Ka);

        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非S2的第一次循环直接复用Q
        inputLeftBuf = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    }

    // 加载当前轮的右矩阵到L1
    if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) { // orikv阶段不需要取topk, 核内同步
        WaitFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);
        LocalTensor<KV_T> inputRightTensor = inputRightBuf.GetTensor<KV_T>();
        if constexpr (IS_PA) {
            Position startPos;
            startPos.bIdx = runInfo.boIdx;
            startPos.n2Idx = runInfo.n2oIdx;
            startPos.s2Offset = coordInfo[runInfo.taskIdMod3].s2Coord;
            startPos.dIdx = 0;
            PAShape shape;
            shape.blockSize = kvCacheBlockSize;
            shape.headNum = constInfo.n2Size;
            shape.headDim = constInfo.dSize;
            shape.actHeadDim = constInfo.dSize;
            shape.maxblockNumPerBatch = maxBlockNumPerBatch;
            shape.copyRowNum = runInfo.s2RealSize;
            shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
            GmCopyInToL1PA<KV_T>(inputRightTensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
        } else {
            int64_t keyOffset = this->curKvGm.offsetCalculator.GetOffset(
                coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
            CopyToL1Nd2Nz<KV_T>(inputRightTensor, curKvGm.gmTensor[keyOffset], runInfo.s2RealSize,
                                constInfo.dSize, constInfo.mm1Kb);
        }
        SetFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
        WaitFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
    } else {
        inputRightBuf.WaitCrossCore();
        WaitFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);
        LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
        v0ResGm.WaitCrossCore();
        GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
        CopyToL1Nd2Nz<Q_T>(dst, v0ResGmTensor, runInfo.s2RealSize, constInfo.dSize, constInfo.mm1Kb);
        SetFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
        WaitFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
    }

    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                        static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                        static_cast<uint32_t>(constInfo.dSize),   // singleK
                        0,    // isLeftTranspose
                        1     // isRightTranspose
                    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(  // m,n不切，k切128
        inputLeftBuf.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(),                // mm1B直接用tensor的数据
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        inputLeftBuf.Set<HardEvent::MTE1_MTE2>(); // 释放L1A
    }

    mm1ResL0C.Set<HardEvent::M_FIX>();    // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>();   // 等待L0C

    outputBuf.WaitCrossCore();
    // L0C→UB
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dstStride = s2BaseSize;
    // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    // 将matmul结果从L0C搬运到UB
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams);
    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2SCFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BACKWARD> inputLeftBuf = inputLeftBuffers.Get(); // P直接用无需搬运
    inputLeftBuf.WaitCrossCore();

    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {static_cast<uint32_t>(s1BaseSize),          // singleM 64
                     static_cast<uint32_t>(constInfo.dSizeV), // singleN 512
                     static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(),
        inputRightBuf.GetTensor<Q_T>(),
        mmL0ABuffers,
        mmL0BBuffers,
        mm2ResL0C.GetTensor<T>(),
        param);

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);
    }

    mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

    // 占用
    outputBuf.WaitCrossCore();
    // L0C→UB;FixpipeParamsM300:L0C→UB
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.mSize = s1BaseSize;
    // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.srcStride = Align16Func(s1BaseSize);
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(
        outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class SCFABlockCubeDummy {
public:
    __aicore__ inline SCFABlockCubeDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe,
        BufferManager<BufferType::L1> &l1BufferManager, __gm__ uint8_t *query) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv,
        __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
        __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
        __gm__ uint8_t *cuSeqlensCmpKv,
        __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, const ConstInfo& constInfo) {}
};

template <typename T>
struct CubeBlockTraits;  // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
