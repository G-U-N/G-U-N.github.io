"""Runnable Triton kernels for the KV-cache article.

The file changes one systems axis at a time:

1. ``flash_gqa_attention`` maps query heads to shared KV heads during training,
   including the per-query-head dK/dV partials and grouped reduction in backward.
2. ``grouped_decode`` packs query heads that consume the same contiguous KV tile.
3. ``paged_grouped_decode`` changes only the KV addressing through a block table.
4. ``mla_decode`` reuses the grouped decoder with a latent score block, a separate
   RoPE score block, and an independently sized value/output block.

The implementation is intentionally compact rather than production tuned.  The
algorithmic references were pinned while the article was written:

* FlagAttention 8225e615ffec19a5481779806a03b134ff4a3b28
* vLLM         5f213ed1592903b7bc38f173d320dac1b2769303
* FlashMLA     15f13e5030374295491c5ce31b02d7e63a7772c6

Run correctness checks on an NVIDIA GPU with:

    python blogs/code/kv_cache_attention.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _flash_gqa_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    M: tl.constexpr,
    N: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DQK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_block = tl.program_id(2)

    # The only semantic change from the Part I MHA kernel: Q/O keep the
    # query-head index, while K/V use the shared KV-head index.
    kv_head = query_head // GROUP_SIZE

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_qk = tl.arange(0, BLOCK_DQK)
    offs_v = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < M
    mask_qk = offs_qk < D_QK
    mask_v = offs_v < D_V

    q_ptrs = (
        Q
        + batch * stride_qb
        + query_head * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_qk[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_qk[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < N
        k_ptrs = (
            K
            + batch * stride_kb
            + kv_head * stride_kh
            + offs_n[None, :] * stride_kn
            + offs_qk[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + batch * stride_vb
            + kv_head * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_v[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=mask_qk[:, None] & mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_v[None, :], other=0.0)

        scores = tl.dot(q, k) * SCALE
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        if CAUSAL:
            # Bottom-right alignment also covers a cached prefix when N > M.
            q_position = N - M + offs_m
            scores = tl.where(q_position[:, None] >= offs_n[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    o_ptrs = (
        O
        + batch * stride_ob
        + query_head * stride_oh
        + offs_m[:, None] * stride_om
        + offs_v[None, :] * stride_od
    )
    l_ptrs = LSE + batch * stride_lb + query_head * stride_lh + offs_m * stride_lm
    tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_v[None, :])
    tl.store(l_ptrs, lse, mask=mask_m)


@triton.jit
def _flash_gqa_bwd_preprocess_kernel(
    O,
    DO,
    DELTA,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_db,
    stride_dh,
    stride_dm,
    M: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_block = tl.program_id(2)

    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_v = tl.arange(0, BLOCK_DV)
    mask = (rows[:, None] < M) & (offs_v[None, :] < D_V)
    o = tl.load(
        O
        + batch * stride_ob
        + query_head * stride_oh
        + rows[:, None] * stride_om
        + offs_v[None, :] * stride_od,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + batch * stride_dob
        + query_head * stride_doh
        + rows[:, None] * stride_dom
        + offs_v[None, :] * stride_dod,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(
        DELTA + batch * stride_db + query_head * stride_dh + rows * stride_dm,
        delta,
        mask=rows < M,
    )


@triton.jit
def _flash_gqa_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    PARTIAL_DK,
    PARTIAL_DV,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_pdkb,
    stride_pdkh,
    stride_pdkn,
    stride_pdkd,
    stride_pdvb,
    stride_pdvh,
    stride_pdvn,
    stride_pdvd,
    M: tl.constexpr,
    N: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DQK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_block = tl.program_id(2)
    kv_head = query_head // GROUP_SIZE

    cols = kv_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_qk = tl.arange(0, BLOCK_DQK)
    offs_v = tl.arange(0, BLOCK_DV)
    mask_k = (cols[:, None] < N) & (offs_qk[None, :] < D_QK)
    mask_v = (cols[:, None] < N) & (offs_v[None, :] < D_V)
    k = tl.load(
        K
        + batch * stride_kb
        + kv_head * stride_kh
        + cols[:, None] * stride_kn
        + offs_qk[None, :] * stride_kd,
        mask=mask_k,
        other=0.0,
    )
    v = tl.load(
        V
        + batch * stride_vb
        + kv_head * stride_vh
        + cols[:, None] * stride_vn
        + offs_v[None, :] * stride_vd,
        mask=mask_v,
        other=0.0,
    )
    dk = tl.zeros([BLOCK_N, BLOCK_DQK], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DV], tl.float32)
    rows_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, M, BLOCK_M):
        rows = start_m + rows_base
        q = tl.load(
            Q
            + batch * stride_qb
            + query_head * stride_qh
            + rows[:, None] * stride_qm
            + offs_qk[None, :] * stride_qd,
            mask=(rows[:, None] < M) & (offs_qk[None, :] < D_QK),
            other=0.0,
        )
        do = tl.load(
            DO
            + batch * stride_dob
            + query_head * stride_doh
            + rows[:, None] * stride_dom
            + offs_v[None, :] * stride_dod,
            mask=(rows[:, None] < M) & (offs_v[None, :] < D_V),
            other=0.0,
        )
        lse = tl.load(
            LSE + batch * stride_lb + query_head * stride_lh + rows * stride_lm,
            mask=rows < M,
            other=0.0,
        )
        delta = tl.load(
            DELTA + batch * stride_db + query_head * stride_dh + rows * stride_dm,
            mask=rows < M,
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * SCALE
        visible = (rows[:, None] < M) & (cols[None, :] < N)
        if CAUSAL:
            q_position = N - M + rows
            visible = visible & (q_position[:, None] >= cols[None, :])
        scores = tl.where(visible, scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        p = tl.where(visible, p, 0.0)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None])

        dv += tl.dot(tl.trans(p.to(DO.dtype.element_ty)), do)
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q) * SCALE

    tl.store(
        PARTIAL_DK
        + batch * stride_pdkb
        + query_head * stride_pdkh
        + cols[:, None] * stride_pdkn
        + offs_qk[None, :] * stride_pdkd,
        dk,
        mask=mask_k,
    )
    tl.store(
        PARTIAL_DV
        + batch * stride_pdvb
        + query_head * stride_pdvh
        + cols[:, None] * stride_pdvn
        + offs_v[None, :] * stride_pdvd,
        dv,
        mask=mask_v,
    )


@triton.jit
def _flash_gqa_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    M: tl.constexpr,
    N: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DQK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_block = tl.program_id(2)
    kv_head = query_head // GROUP_SIZE

    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_qk = tl.arange(0, BLOCK_DQK)
    offs_v = tl.arange(0, BLOCK_DV)
    q = tl.load(
        Q
        + batch * stride_qb
        + query_head * stride_qh
        + rows[:, None] * stride_qm
        + offs_qk[None, :] * stride_qd,
        mask=(rows[:, None] < M) & (offs_qk[None, :] < D_QK),
        other=0.0,
    )
    do = tl.load(
        DO
        + batch * stride_dob
        + query_head * stride_doh
        + rows[:, None] * stride_dom
        + offs_v[None, :] * stride_dod,
        mask=(rows[:, None] < M) & (offs_v[None, :] < D_V),
        other=0.0,
    )
    lse = tl.load(
        LSE + batch * stride_lb + query_head * stride_lh + rows * stride_lm,
        mask=rows < M,
        other=0.0,
    )
    delta = tl.load(
        DELTA + batch * stride_db + query_head * stride_dh + rows * stride_dm,
        mask=rows < M,
        other=0.0,
    )
    dq = tl.zeros([BLOCK_M, BLOCK_DQK], tl.float32)
    cols_base = tl.arange(0, BLOCK_N)

    for start_n in range(0, N, BLOCK_N):
        cols = start_n + cols_base
        k = tl.load(
            K
            + batch * stride_kb
            + kv_head * stride_kh
            + cols[:, None] * stride_kn
            + offs_qk[None, :] * stride_kd,
            mask=(cols[:, None] < N) & (offs_qk[None, :] < D_QK),
            other=0.0,
        )
        v = tl.load(
            V
            + batch * stride_vb
            + kv_head * stride_vh
            + cols[:, None] * stride_vn
            + offs_v[None, :] * stride_vd,
            mask=(cols[:, None] < N) & (offs_v[None, :] < D_V),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * SCALE
        visible = (rows[:, None] < M) & (cols[None, :] < N)
        if CAUSAL:
            q_position = N - M + rows
            visible = visible & (q_position[:, None] >= cols[None, :])
        scores = tl.where(visible, scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        p = tl.where(visible, p, 0.0)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(Q.dtype.element_ty), k) * SCALE

    tl.store(
        DQ
        + batch * stride_dqb
        + query_head * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_qk[None, :] * stride_dqd,
        dq,
        mask=(rows[:, None] < M) & (offs_qk[None, :] < D_QK),
    )


@triton.jit
def _grouped_decode_kernel(
    Q,
    K,
    V,
    SEQ_LENS,
    O,
    LSE,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    N: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DQK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    head_tile = tl.program_id(1)

    # program_id(1) owns a tile of query heads inside exactly one KV group.
    tiles_per_kv = tl.cdiv(GROUP_SIZE, BLOCK_H)
    kv_head = head_tile // tiles_per_kv
    tile_in_group = head_tile - kv_head * tiles_per_kv
    query_heads = (
        kv_head * GROUP_SIZE
        + tile_in_group * BLOCK_H
        + tl.arange(0, BLOCK_H)
    )
    mask_h = (
        (query_heads < (kv_head + 1) * GROUP_SIZE)
        & (query_heads < H_Q)
        & (kv_head < H_KV)
    )

    offs_qk = tl.arange(0, BLOCK_DQK)
    offs_v = tl.arange(0, BLOCK_DV)
    mask_qk = offs_qk < D_QK
    mask_v = offs_v < D_V

    q_ptrs = (
        Q
        + batch * stride_qb
        + query_heads[:, None] * stride_qh
        + offs_qk[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_h[:, None] & mask_qk[None, :], other=0.0)

    valid_n = tl.load(SEQ_LENS + batch)
    m_i = tl.where(
        mask_h,
        tl.full([BLOCK_H], -float("inf"), tl.float32),
        0.0,
    )
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], tl.float32)
    offs_n_base = tl.arange(0, BLOCK_N)

    # One program loads each KV tile once, then uses it for BLOCK_H queries.
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < valid_n
        k_ptrs = (
            K
            + batch * stride_kb
            + kv_head * stride_kh
            + offs_n[None, :] * stride_kn
            + offs_qk[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + batch * stride_vb
            + kv_head * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_v[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=mask_qk[:, None] & mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_v[None, :], other=0.0)

        scores = tl.dot(q, k) * SCALE
        scores = tl.where(mask_h[:, None] & mask_n[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    o_ptrs = (
        O
        + batch * stride_ob
        + query_heads[:, None] * stride_oh
        + offs_v[None, :] * stride_od
    )
    l_ptrs = LSE + batch * stride_lb + query_heads * stride_lh
    tl.store(o_ptrs, out, mask=mask_h[:, None] & mask_v[None, :])
    tl.store(l_ptrs, lse, mask=mask_h)


@triton.jit
def _grouped_decode_split_kernel(
    Q,
    K,
    V,
    Q_PE,
    K_PE,
    SEQ_LENS,
    PARTIAL_O,
    PARTIAL_LSE,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_qpeb,
    stride_qpeh,
    stride_qped,
    stride_kpeb,
    stride_kpeh,
    stride_kpen,
    stride_kped,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_plb,
    stride_plh,
    stride_pls,
    N: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    D_CONTENT: tl.constexpr,
    D_PE: tl.constexpr,
    D_V: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    head_tile = tl.program_id(1)
    split = tl.program_id(2)

    tiles_per_kv = tl.cdiv(GROUP_SIZE, BLOCK_H)
    kv_head = head_tile // tiles_per_kv
    tile_in_group = head_tile - kv_head * tiles_per_kv
    query_heads = kv_head * GROUP_SIZE + tile_in_group * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (query_heads < (kv_head + 1) * GROUP_SIZE) & (query_heads < H_Q) & (kv_head < H_KV)

    offs_dc = tl.arange(0, BLOCK_DC)
    offs_dpe = tl.arange(0, BLOCK_DPE)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dc = offs_dc < D_CONTENT
    mask_dpe = offs_dpe < D_PE
    mask_dv = offs_dv < D_V

    q_ptrs = Q + batch * stride_qb + query_heads[:, None] * stride_qh + offs_dc[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_h[:, None] & mask_dc[None, :], other=0.0)
    if D_PE > 0:
        qpe_ptrs = (
            Q_PE
            + batch * stride_qpeb
            + query_heads[:, None] * stride_qpeh
            + offs_dpe[None, :] * stride_qped
        )
        q_pe = tl.load(qpe_ptrs, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0)

    valid_n = tl.load(SEQ_LENS + batch)
    blocks_per_split = tl.cdiv(tl.cdiv(N, NUM_SPLITS), BLOCK_N)
    split_size = blocks_per_split * BLOCK_N
    split_start = split * split_size
    split_end = tl.minimum(split_start + split_size, valid_n)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], tl.float32)
    offs_n_base = tl.arange(0, BLOCK_N)

    for start_n in range(split_start, split_end, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < split_end
        k_ptrs = (
            K
            + batch * stride_kb
            + kv_head * stride_kh
            + offs_n[None, :] * stride_kn
            + offs_dc[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + batch * stride_vb
            + kv_head * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_dv[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=mask_dc[:, None] & mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        scores = tl.dot(q, k) * SCALE
        if D_PE > 0:
            kpe_ptrs = (
                K_PE
                + batch * stride_kpeb
                + kv_head * stride_kpeh
                + offs_n[None, :] * stride_kpen
                + offs_dpe[:, None] * stride_kped
            )
            k_pe = tl.load(kpe_ptrs, mask=mask_dpe[:, None] & mask_n[None, :], other=0.0)
            scores = scores + tl.dot(q_pe, k_pe) * SCALE
        scores = tl.where(mask_h[:, None] & mask_n[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2((m_i - m_new) * _LOG2E)
        p = tl.exp2((scores - m_new[:, None]) * _LOG2E)
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    has_tokens = l_i > 0.0
    partial_o = tl.where(has_tokens[:, None], acc / l_i[:, None], 0.0)
    partial_lse = tl.where(has_tokens, m_i + tl.log(l_i), -float("inf"))
    po_ptrs = (
        PARTIAL_O
        + batch * stride_pob
        + query_heads[:, None] * stride_poh
        + split * stride_pos
        + offs_dv[None, :] * stride_pod
    )
    pl_ptrs = PARTIAL_LSE + batch * stride_plb + query_heads * stride_plh + split * stride_pls
    tl.store(po_ptrs, partial_o, mask=mask_h[:, None] & mask_dv[None, :])
    tl.store(pl_ptrs, partial_lse, mask=mask_h)


@triton.jit
def _paged_grouped_decode_split_kernel(
    Q,
    K,
    V,
    Q_PE,
    K_PE,
    BLOCK_TABLE,
    SEQ_LENS,
    PARTIAL_O,
    PARTIAL_LSE,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vp,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_qpeb,
    stride_qpeh,
    stride_qped,
    stride_kpep,
    stride_kpet,
    stride_kpeh,
    stride_kped,
    stride_btb,
    stride_btn,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_plb,
    stride_plh,
    stride_pls,
    N: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    D_CONTENT: tl.constexpr,
    D_PE: tl.constexpr,
    D_V: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    head_tile = tl.program_id(1)
    split = tl.program_id(2)

    tiles_per_kv = tl.cdiv(GROUP_SIZE, BLOCK_H)
    kv_head = head_tile // tiles_per_kv
    tile_in_group = head_tile - kv_head * tiles_per_kv
    query_heads = kv_head * GROUP_SIZE + tile_in_group * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (query_heads < (kv_head + 1) * GROUP_SIZE) & (query_heads < H_Q) & (kv_head < H_KV)

    offs_dc = tl.arange(0, BLOCK_DC)
    offs_dpe = tl.arange(0, BLOCK_DPE)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dc = offs_dc < D_CONTENT
    mask_dpe = offs_dpe < D_PE
    mask_dv = offs_dv < D_V
    q = tl.load(
        Q + batch * stride_qb + query_heads[:, None] * stride_qh + offs_dc[None, :] * stride_qd,
        mask=mask_h[:, None] & mask_dc[None, :],
        other=0.0,
    )
    if D_PE > 0:
        q_pe = tl.load(
            Q_PE + batch * stride_qpeb + query_heads[:, None] * stride_qpeh + offs_dpe[None, :] * stride_qped,
            mask=mask_h[:, None] & mask_dpe[None, :],
            other=0.0,
        )

    valid_n = tl.load(SEQ_LENS + batch)
    blocks_per_split = tl.cdiv(tl.cdiv(N, NUM_SPLITS), BLOCK_N)
    split_size = blocks_per_split * BLOCK_N
    split_start = split * split_size
    split_end = tl.minimum(split_start + split_size, valid_n)
    offs_n_base = tl.arange(0, BLOCK_N)
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], tl.float32)

    for start_n in range(split_start, split_end, BLOCK_N):
        logical_n = start_n + offs_n_base
        mask_n = logical_n < split_end
        logical_page = logical_n // PAGE_SIZE
        page_offset = logical_n - logical_page * PAGE_SIZE
        physical_page = tl.load(
            BLOCK_TABLE + batch * stride_btb + logical_page * stride_btn,
            mask=mask_n,
            other=0,
        )
        k = tl.load(
            K
            + physical_page[None, :] * stride_kp
            + page_offset[None, :] * stride_kt
            + kv_head * stride_kh
            + offs_dc[:, None] * stride_kd,
            mask=mask_dc[:, None] & mask_n[None, :],
            other=0.0,
        )
        v = tl.load(
            V
            + physical_page[:, None] * stride_vp
            + page_offset[:, None] * stride_vt
            + kv_head * stride_vh
            + offs_dv[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_dv[None, :],
            other=0.0,
        )
        scores = tl.dot(q, k) * SCALE
        if D_PE > 0:
            k_pe = tl.load(
                K_PE
                + physical_page[None, :] * stride_kpep
                + page_offset[None, :] * stride_kpet
                + kv_head * stride_kpeh
                + offs_dpe[:, None] * stride_kped,
                mask=mask_dpe[:, None] & mask_n[None, :],
                other=0.0,
            )
            scores = scores + tl.dot(q_pe, k_pe) * SCALE
        scores = tl.where(mask_h[:, None] & mask_n[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2((m_i - m_new) * _LOG2E)
        p = tl.exp2((scores - m_new[:, None]) * _LOG2E)
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    has_tokens = l_i > 0.0
    partial_o = tl.where(has_tokens[:, None], acc / l_i[:, None], 0.0)
    partial_lse = tl.where(has_tokens, m_i + tl.log(l_i), -float("inf"))
    po_ptrs = (
        PARTIAL_O
        + batch * stride_pob
        + query_heads[:, None] * stride_poh
        + split * stride_pos
        + offs_dv[None, :] * stride_pod
    )
    pl_ptrs = PARTIAL_LSE + batch * stride_plb + query_heads * stride_plh + split * stride_pls
    tl.store(po_ptrs, partial_o, mask=mask_h[:, None] & mask_dv[None, :])
    tl.store(pl_ptrs, partial_lse, mask=mask_h)


@triton.jit
def _combine_splits_kernel(
    PARTIAL_O,
    PARTIAL_LSE,
    O,
    LSE,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_plb,
    stride_plh,
    stride_pls,
    stride_ob,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    D_V: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < D_V

    max_lse = -float("inf")
    for split in range(0, NUM_SPLITS):
        part_lse = tl.load(PARTIAL_LSE + batch * stride_plb + head * stride_plh + split * stride_pls)
        max_lse = tl.maximum(max_lse, part_lse)

    denom = 0.0
    acc = tl.zeros([BLOCK_DV], tl.float32)
    for split in range(0, NUM_SPLITS):
        part_lse = tl.load(PARTIAL_LSE + batch * stride_plb + head * stride_plh + split * stride_pls)
        weight = tl.exp(part_lse - max_lse)
        part_o = tl.load(
            PARTIAL_O
            + batch * stride_pob
            + head * stride_poh
            + split * stride_pos
            + offs_dv * stride_pod,
            mask=mask_dv,
            other=0.0,
        )
        denom += weight
        acc += weight * part_o

    out = acc / denom
    final_lse = max_lse + tl.log(denom)
    tl.store(O + batch * stride_ob + head * stride_oh + offs_dv * stride_od, out, mask=mask_dv)
    tl.store(LSE + batch * stride_lb + head * stride_lh, final_lse)


def _power_of_two_block(size: int, minimum: int = 16) -> int:
    return max(minimum, triton.next_power_of_2(size))


def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("prefill q, k, and v must have shape [B, H, S, D]")
    b, hq, m, dqk = q.shape
    bk, hkv, n, dk = k.shape
    bv, hv, nv, dv = v.shape
    if (b, hkv, n) != (bk, hv, nv) or b != bv or dqk != dk:
        raise ValueError("q/k feature widths and k/v batch, head, and sequence shapes must agree")
    if hq % hkv:
        raise ValueError("H_q must be divisible by H_kv")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, and V must use the same dtype in this teaching kernel")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("the Triton kernels require CUDA tensors")
    return b, hq, hkv, m, n, dv


def flash_gqa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-style prefill for MHA, GQA, or MQA."""
    b, hq, hkv, m, n, dv = _check_qkv(q, k, v)
    if causal and m > n:
        raise ValueError("bottom-right causal alignment requires S_q <= S_kv")
    dqk = q.shape[-1]
    scale = float(scale if scale is not None else 1.0 / math.sqrt(dqk))
    out = torch.empty((b, hq, m, dv), device=q.device, dtype=v.dtype)
    lse = torch.empty((b, hq, m), device=q.device, dtype=torch.float32)
    block_m, block_n = 64, 64
    grid = (b, hq, triton.cdiv(m, block_m))
    _flash_gqa_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *lse.stride(),
        M=m,
        N=n,
        D_QK=dqk,
        D_V=dv,
        GROUP_SIZE=hq // hkv,
        SCALE=scale,
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DQK=_power_of_two_block(dqk),
        BLOCK_DV=_power_of_two_block(dv),
        num_warps=4,
        num_stages=2,
    )
    return out, lse


def reduce_grouped_kv_grads(partial_dk: torch.Tensor, partial_dv: torch.Tensor, hkv: int):
    """Reduce per-query-head dK/dV contributions into shared KV heads."""
    b, hq = partial_dk.shape[:2]
    if hq % hkv:
        raise ValueError("H_q must be divisible by H_kv")
    group = hq // hkv
    dk = partial_dk.reshape(b, hkv, group, *partial_dk.shape[2:]).sum(dim=2)
    dv = partial_dv.reshape(b, hkv, group, *partial_dv.shape[2:]).sum(dim=2)
    return dk, dv


def flash_gqa_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlagAttention-style full-sequence GQA backward.

    dK and dV are first produced per query head, then reduced across the query
    heads that share each KV head.  This intentionally mirrors FlagAttention's
    simple, readable ownership scheme rather than a fused production reduction.
    """
    b, hq, hkv, m, n, dv = _check_qkv(q, k, v)
    if causal and m > n:
        raise ValueError("bottom-right causal alignment requires S_q <= S_kv")
    dqk = q.shape[-1]
    expected_o = (b, hq, m, dv)
    if out.shape != expected_o or do.shape != expected_o:
        raise ValueError(f"out and do must have shape {expected_o}")
    if lse.shape != (b, hq, m) or lse.dtype != torch.float32:
        raise ValueError("lse must be a float32 tensor with shape [B, H_q, M]")
    if out.device != q.device or do.device != q.device or lse.device != q.device:
        raise ValueError("out, do, and lse must be on the same device as Q")
    if out.dtype != v.dtype or do.dtype != v.dtype:
        raise ValueError("out and do must use V's dtype")

    scale = float(scale if scale is not None else 1.0 / math.sqrt(dqk))
    delta = torch.empty((b, hq, m), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    partial_dk = torch.empty((b, hq, n, dqk), device=k.device, dtype=k.dtype)
    partial_dv = torch.empty((b, hq, n, dv), device=v.device, dtype=v.dtype)

    block_m, block_n = 64, 64
    block_dqk = _power_of_two_block(dqk)
    block_dv = _power_of_two_block(dv)
    preprocess_grid = (b, hq, triton.cdiv(m, block_m))
    _flash_gqa_bwd_preprocess_kernel[preprocess_grid](
        out,
        do,
        delta,
        *out.stride(),
        *do.stride(),
        *delta.stride(),
        M=m,
        D_V=dv,
        BLOCK_M=block_m,
        BLOCK_DV=block_dv,
        num_warps=4,
    )

    dkdv_grid = (b, hq, triton.cdiv(n, block_n))
    _flash_gqa_bwd_dkdv_kernel[dkdv_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        partial_dk,
        partial_dv,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
        *lse.stride(),
        *delta.stride(),
        *partial_dk.stride(),
        *partial_dv.stride(),
        M=m,
        N=n,
        D_QK=dqk,
        D_V=dv,
        GROUP_SIZE=hq // hkv,
        SCALE=scale,
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DQK=block_dqk,
        BLOCK_DV=block_dv,
        num_warps=4,
        num_stages=2,
    )

    dq_grid = (b, hq, triton.cdiv(m, block_m))
    _flash_gqa_bwd_dq_kernel[dq_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
        *lse.stride(),
        *delta.stride(),
        *dq.stride(),
        M=m,
        N=n,
        D_QK=dqk,
        D_V=dv,
        GROUP_SIZE=hq // hkv,
        SCALE=scale,
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DQK=block_dqk,
        BLOCK_DV=block_dv,
        num_warps=4,
        num_stages=2,
    )
    dk, dv_out = reduce_grouped_kv_grads(partial_dk, partial_dv, hkv)
    return dq, dk, dv_out


class _FlashGQAAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal):
        scale_value = float(scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
        out, lse = flash_gqa_forward(q, k, v, scale=scale_value, causal=causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale_value
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = flash_gqa_backward(
            q,
            k,
            v,
            out,
            lse,
            do,
            scale=ctx.scale,
            causal=ctx.causal,
        )
        return dq, dk, dv, None, None


def flash_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Differentiable full-sequence MHA/GQA/MQA attention."""
    return _FlashGQAAttention.apply(q, k, v, scale, causal)


@dataclass(frozen=True)
class _DecodeShape:
    batch: int
    hq: int
    hkv: int
    n: int
    dc: int
    dpe: int
    dv: int

    @property
    def group_size(self) -> int:
        return self.hq // self.hkv


def _prepare_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rope: torch.Tensor | None,
    k_rope: torch.Tensor | None,
    *,
    paged: bool,
) -> tuple[_DecodeShape, torch.Tensor, torch.Tensor]:
    if q.ndim != 3:
        raise ValueError("decode q must have shape [B, H_q, D_content]")
    expected_ndim = 4
    if k.ndim != expected_ndim or v.ndim != expected_ndim:
        layout = "[pages, page_size, H_kv, D]" if paged else "[B, H_kv, S_kv, D]"
        raise ValueError(f"decode caches must have shape {layout}")
    b, hq, dc = q.shape
    if paged:
        hkv, n = k.shape[2], -1
        if k.shape[:3] != v.shape[:3] or k.shape[-1] != dc:
            raise ValueError("paged K and V must share page, token, and head axes")
    else:
        bk, hkv, n, dk = k.shape
        if bk != b or dk != dc or v.shape[:3] != (b, hkv, n):
            raise ValueError("contiguous K/V shapes do not match Q")
    if hq % hkv:
        raise ValueError("H_q must be divisible by H_kv")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, and V must use the same dtype in this teaching kernel")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("the Triton kernels require CUDA tensors")

    if (q_rope is None) != (k_rope is None):
        raise ValueError("q_rope and k_rope must be supplied together")
    if q_rope is None:
        q_rope = torch.empty((b, hq, 1), device=q.device, dtype=q.dtype)
        if paged:
            k_rope = torch.empty((*k.shape[:3], 1), device=k.device, dtype=k.dtype)
        else:
            k_rope = torch.empty((b, hkv, n, 1), device=k.device, dtype=k.dtype)
        dpe = 0
    else:
        dpe = q_rope.shape[-1]
        if q_rope.ndim != 3 or q_rope.shape[:2] != (b, hq):
            raise ValueError("q_rope must have shape [B, H_q, D_rope]")
        expected = (*k.shape[:3], dpe)
        if k_rope.shape != expected:
            raise ValueError(f"k_rope must have shape {expected}")
        if q_rope.dtype != q.dtype or k_rope.dtype != k.dtype:
            raise ValueError("the rotary Q/K blocks must use the Q/K dtype")
        if q_rope.device != q.device or k_rope.device != k.device:
            raise ValueError("the rotary Q/K blocks must be on the Q/K device")

    dv = v.shape[-1]
    shape = _DecodeShape(b, hq, hkv, n, dc, dpe, dv)
    return shape, q_rope, k_rope


def _allocate_partials(shape: _DecodeShape, num_splits: int, dtype: torch.dtype, device: torch.device):
    partial_o = torch.empty((shape.batch, shape.hq, num_splits, shape.dv), device=device, dtype=dtype)
    partial_lse = torch.empty((shape.batch, shape.hq, num_splits), device=device, dtype=torch.float32)
    out = torch.empty((shape.batch, shape.hq, shape.dv), device=device, dtype=dtype)
    lse = torch.empty((shape.batch, shape.hq), device=device, dtype=torch.float32)
    return partial_o, partial_lse, out, lse


def _launch_combine(partial_o, partial_lse, out, lse, shape: _DecodeShape, num_splits: int):
    _combine_splits_kernel[(shape.batch, shape.hq)](
        partial_o,
        partial_lse,
        out,
        lse,
        *partial_o.stride(),
        *partial_lse.stride(),
        *out.stride(),
        *lse.stride(),
        D_V=shape.dv,
        NUM_SPLITS=num_splits,
        BLOCK_DV=_power_of_two_block(shape.dv),
        num_warps=4,
    )


def grouped_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_splits: int = 1,
    q_rope: torch.Tensor | None = None,
    k_rope: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token grouped decode over a contiguous cache."""
    has_separate_rope = q_rope is not None
    shape, q_rope, k_rope = _prepare_decode_inputs(q, k, v, q_rope, k_rope, paged=False)
    if seq_lens.shape != (shape.batch,):
        raise ValueError("seq_lens must have shape [B]")
    if seq_lens.device != q.device or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be an int32 tensor on the same device as Q")
    if bool(torch.any(seq_lens <= 0).item()) or bool(torch.any(seq_lens > shape.n).item()):
        raise ValueError("every sequence length must lie in [1, S_kv]")
    if num_splits < 1:
        raise ValueError("num_splits must be positive")
    block_h = 16
    block_n = 32 if max(shape.dc, shape.dv) >= 256 else 64
    num_stages = 1 if max(shape.dc, shape.dv) >= 256 else 2
    head_tiles = shape.hkv * triton.cdiv(shape.group_size, block_h)

    # The unsplit GQA/MQA path makes the full-sequence-versus-decode change
    # explicit: query rows become the dot-product M dimension, and one program
    # streams the entire KV sequence. MLA keeps using the generalized kernel
    # below because it has a separate RoPE score block.
    if num_splits == 1 and not has_separate_rope:
        out = torch.empty((shape.batch, shape.hq, shape.dv), device=v.device, dtype=v.dtype)
        lse = torch.empty((shape.batch, shape.hq), device=q.device, dtype=torch.float32)
        grid = (shape.batch, head_tiles)
        _grouped_decode_kernel[grid](
            q,
            k,
            v,
            seq_lens,
            out,
            lse,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            *lse.stride(),
            N=shape.n,
            H_Q=shape.hq,
            H_KV=shape.hkv,
            GROUP_SIZE=shape.group_size,
            D_QK=shape.dc,
            D_V=shape.dv,
            SCALE=float(scale),
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            BLOCK_DQK=_power_of_two_block(shape.dc),
            BLOCK_DV=_power_of_two_block(shape.dv),
            num_warps=4,
            num_stages=num_stages,
        )
        return out, lse

    partial_o, partial_lse, out, lse = _allocate_partials(shape, num_splits, v.dtype, v.device)
    grid = (shape.batch, head_tiles, num_splits)
    _grouped_decode_split_kernel[grid](
        q,
        k,
        v,
        q_rope,
        k_rope,
        seq_lens,
        partial_o,
        partial_lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *q_rope.stride(),
        *k_rope.stride(),
        *partial_o.stride(),
        *partial_lse.stride(),
        N=shape.n,
        H_Q=shape.hq,
        H_KV=shape.hkv,
        GROUP_SIZE=shape.group_size,
        D_CONTENT=shape.dc,
        D_PE=shape.dpe,
        D_V=shape.dv,
        NUM_SPLITS=num_splits,
        SCALE=float(scale),
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_DC=_power_of_two_block(shape.dc),
        BLOCK_DPE=_power_of_two_block(shape.dpe),
        BLOCK_DV=_power_of_two_block(shape.dv),
        num_warps=4,
        num_stages=num_stages,
    )
    _launch_combine(partial_o, partial_lse, out, lse, shape, num_splits)
    return out, lse


def paged_grouped_decode(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_splits: int = 1,
    q_rope: torch.Tensor | None = None,
    k_rope_pages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same grouped split-KV decoder with paged cache loads."""
    shape, q_rope, k_rope_pages = _prepare_decode_inputs(
        q, k_pages, v_pages, q_rope, k_rope_pages, paged=True
    )
    if block_table.ndim != 2 or block_table.shape[0] != shape.batch:
        raise ValueError("block_table must have shape [B, max_logical_pages]")
    if seq_lens.shape != (shape.batch,):
        raise ValueError("seq_lens must have shape [B]")
    if block_table.device != q.device or block_table.dtype != torch.int32:
        raise ValueError("block_table must be an int32 tensor on the same device as Q")
    if seq_lens.device != q.device or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be an int32 tensor on the same device as Q")
    if bool(torch.any(seq_lens <= 0).item()):
        raise ValueError("every sequence length must be positive")
    if num_splits < 1:
        raise ValueError("num_splits must be positive")
    n = int(seq_lens.max().item())
    required_pages = triton.cdiv(n, k_pages.shape[1])
    if block_table.shape[1] < required_pages:
        raise ValueError("block_table does not cover the longest sequence")
    used_pages = block_table[:, :required_pages]
    page_counts = (seq_lens + k_pages.shape[1] - 1) // k_pages.shape[1]
    logical_pages = torch.arange(required_pages, device=q.device)[None, :]
    physical_pages = used_pages[logical_pages < page_counts[:, None]]
    if bool(torch.any(physical_pages < 0).item()) or bool(torch.any(physical_pages >= k_pages.shape[0]).item()):
        raise ValueError("block_table contains a physical page outside the cache pool")
    shape = _DecodeShape(shape.batch, shape.hq, shape.hkv, n, shape.dc, shape.dpe, shape.dv)
    partial_o, partial_lse, out, lse = _allocate_partials(shape, num_splits, v_pages.dtype, v_pages.device)
    block_h = 16
    block_n = 32 if max(shape.dc, shape.dv) >= 256 else 64
    num_stages = 1 if max(shape.dc, shape.dv) >= 256 else 2
    head_tiles = shape.hkv * triton.cdiv(shape.group_size, block_h)
    grid = (shape.batch, head_tiles, num_splits)
    _paged_grouped_decode_split_kernel[grid](
        q,
        k_pages,
        v_pages,
        q_rope,
        k_rope_pages,
        block_table,
        seq_lens,
        partial_o,
        partial_lse,
        *q.stride(),
        *k_pages.stride(),
        *v_pages.stride(),
        *q_rope.stride(),
        *k_rope_pages.stride(),
        *block_table.stride(),
        *partial_o.stride(),
        *partial_lse.stride(),
        N=n,
        H_Q=shape.hq,
        H_KV=shape.hkv,
        GROUP_SIZE=shape.group_size,
        D_CONTENT=shape.dc,
        D_PE=shape.dpe,
        D_V=shape.dv,
        NUM_SPLITS=num_splits,
        PAGE_SIZE=k_pages.shape[1],
        SCALE=float(scale),
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_DC=_power_of_two_block(shape.dc),
        BLOCK_DPE=_power_of_two_block(shape.dpe),
        BLOCK_DV=_power_of_two_block(shape.dv),
        num_warps=4,
        num_stages=num_stages,
    )
    _launch_combine(partial_o, partial_lse, out, lse, shape, num_splits)
    return out, lse


def mla_decode(
    q_absorbed: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-style absorbed MLA attention over a contiguous latent cache.

    The returned value is the per-query-head latent output.  The head-specific
    value/output up-projection remains outside the attention kernel.
    """
    if c_kv.ndim != 3 or k_rope.ndim != 3:
        raise ValueError("c_kv and k_rope must have shape [B, S_kv, D]")
    shared_c = c_kv[:, None, :, :]
    shared_rope = k_rope[:, None, :, :]
    return grouped_decode(
        q_absorbed,
        shared_c,
        shared_c,
        seq_lens,
        scale=scale,
        num_splits=num_splits,
        q_rope=q_rope,
        k_rope=shared_rope,
    )


def attention_reference(q, k, v, *, scale: float, causal: bool = False):
    """Small PyTorch reference; it deliberately materializes the score matrix."""
    group_size = q.shape[1] // k.shape[1]
    kq = k.repeat_interleave(group_size, dim=1)
    vq = v.repeat_interleave(group_size, dim=1)
    scores = torch.einsum("bhmd,bhnd->bhmn", q.float(), kq.float()) * scale
    if causal:
        m, n = q.shape[2], k.shape[2]
        q_pos = n - m + torch.arange(m, device=q.device)
        k_pos = torch.arange(n, device=q.device)
        scores = scores.masked_fill(q_pos[:, None] < k_pos[None, :], -torch.inf)
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhmn,bhnd->bhmd", p, vq.float())
    return out.to(v.dtype), torch.logsumexp(scores, dim=-1)


def decode_reference(q, k, v, seq_lens, *, scale: float, q_rope=None, k_rope=None):
    group_size = q.shape[1] // k.shape[1]
    kq = k.repeat_interleave(group_size, dim=1)
    vq = v.repeat_interleave(group_size, dim=1)
    scores = torch.einsum("bhd,bhnd->bhn", q.float(), kq.float())
    if q_rope is not None:
        kr = k_rope.repeat_interleave(group_size, dim=1)
        scores += torch.einsum("bhr,bhnr->bhn", q_rope.float(), kr.float())
    scores *= scale
    positions = torch.arange(k.shape[2], device=q.device)
    scores = scores.masked_fill(positions[None, None, :] >= seq_lens[:, None, None], -torch.inf)
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhn,bhnd->bhd", p, vq.float())
    return out.to(v.dtype), torch.logsumexp(scores, dim=-1)


def mla_unabsorbed_reference(q_content, q_rope, c_kv, k_rope, w_uk, w_uv, *, scale: float):
    """Materialize per-head K/V to verify projection absorption."""
    k_content = torch.einsum("bnc,hdc->bhnd", c_kv.float(), w_uk.float())
    values = torch.einsum("bnc,hvc->bhnv", c_kv.float(), w_uv.float())
    scores = torch.einsum("bhd,bhnd->bhn", q_content.float(), k_content)
    scores += torch.einsum("bhr,bnr->bhn", q_rope.float(), k_rope.float())
    p = torch.softmax(scores * scale, dim=-1)
    return torch.einsum("bhn,bhnv->bhv", p, values)


def mla_absorbed_reference(q_content, q_rope, c_kv, k_rope, w_uk, w_uv, *, scale: float):
    q_absorbed = torch.einsum("bhd,hdc->bhc", q_content.float(), w_uk.float())
    scores = torch.einsum("bhc,bnc->bhn", q_absorbed, c_kv.float())
    scores += torch.einsum("bhr,bnr->bhn", q_rope.float(), k_rope.float())
    p = torch.softmax(scores * scale, dim=-1)
    latent = torch.einsum("bhn,bnc->bhc", p, c_kv.float())
    out = torch.einsum("bhc,hvc->bhv", latent, w_uv.float())
    return q_absorbed, latent, out


def _pack_pages(x: torch.Tensor, page_size: int, physical_pages: int):
    b, h, n, d = x.shape
    logical_pages = triton.cdiv(n, page_size)
    if physical_pages < b * logical_pages:
        raise ValueError("not enough physical pages")
    pages = torch.zeros((physical_pages, page_size, h, d), device=x.device, dtype=x.dtype)
    permutation = torch.randperm(physical_pages, device=x.device)
    table = permutation[: b * logical_pages].reshape(b, logical_pages).to(torch.int32)
    for batch in range(b):
        for logical in range(logical_pages):
            start = logical * page_size
            end = min(start + page_size, n)
            pages[table[batch, logical], : end - start] = x[batch, :, start:end].transpose(0, 1)
    return pages, table


def run_correctness_checks(dtype=torch.float16):
    if not torch.cuda.is_available():
        raise RuntimeError("correctness checks require an NVIDIA CUDA GPU")
    torch.manual_seed(0)
    device = "cuda"

    for hq, hkv in ((8, 8), (8, 2), (8, 1)):
        q = torch.randn((2, hq, 17, 64), device=device, dtype=dtype)
        k = torch.randn((2, hkv, 23, 64), device=device, dtype=dtype)
        v = torch.randn((2, hkv, 23, 48), device=device, dtype=dtype)
        got, got_lse = flash_gqa_forward(q, k, v, scale=0.125, causal=True)
        ref, ref_lse = attention_reference(q, k, v, scale=0.125, causal=True)
        torch.testing.assert_close(got, ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(got_lse, ref_lse, atol=3e-2, rtol=3e-2)

    # Full-sequence backward: MHA, GQA, and MQA with D_qk != D_v and M != N.
    grad_atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    for hq, hkv, causal in ((4, 4, False), (4, 2, True), (4, 1, True)):
        q0 = torch.randn((1, hq, 17, 64), device=device, dtype=dtype) * 0.4
        k0 = torch.randn((1, hkv, 23, 64), device=device, dtype=dtype) * 0.4
        v0 = torch.randn((1, hkv, 23, 48), device=device, dtype=dtype) * 0.4
        do = torch.randn((1, hq, 17, 48), device=device, dtype=dtype) * 0.4

        q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_(True) for x in (q0, k0, v0)]
        out_ref, _ = attention_reference(q_ref, k_ref, v_ref, scale=0.125, causal=causal)
        grads_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), do)

        q_tri, k_tri, v_tri = [x.detach().clone().requires_grad_(True) for x in (q0, k0, v0)]
        out_tri = flash_gqa_attention(q_tri, k_tri, v_tri, scale=0.125, causal=causal)
        grads_tri = torch.autograd.grad(out_tri, (q_tri, k_tri, v_tri), do)
        torch.testing.assert_close(out_tri, out_ref, atol=3e-2, rtol=3e-2)
        for got_grad, ref_grad in zip(grads_tri, grads_ref):
            torch.testing.assert_close(got_grad, ref_grad, atol=grad_atol, rtol=grad_atol)

    b, hq, hkv, n, d = 2, 32, 8, 257, 64
    q = torch.randn((b, hq, d), device=device, dtype=dtype)
    k = torch.randn((b, hkv, n, d), device=device, dtype=dtype)
    v = torch.randn((b, hkv, n, 48), device=device, dtype=dtype)
    seq_lens = torch.tensor([257, 219], device=device, dtype=torch.int32)
    ref, ref_lse = decode_reference(q, k, v, seq_lens, scale=0.125)
    for splits in (1, 2, 4):
        got, got_lse = grouped_decode(q, k, v, seq_lens, scale=0.125, num_splits=splits)
        torch.testing.assert_close(got, ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(got_lse, ref_lse, atol=3e-2, rtol=3e-2)

    k_pages, table = _pack_pages(k, 16, physical_pages=40)
    v_pages, table_v = _pack_pages(v, 16, physical_pages=40)
    # Repack V according to K's table so both caches share one block table.
    if not torch.equal(table, table_v):
        v_pages.zero_()
        for batch in range(b):
            for logical in range(table.shape[1]):
                start, end = logical * 16, min((logical + 1) * 16, n)
                v_pages[table[batch, logical], : end - start] = v[batch, :, start:end].transpose(0, 1)
    got, got_lse = paged_grouped_decode(
        q, k_pages, v_pages, table, seq_lens, scale=0.125, num_splits=4
    )
    torch.testing.assert_close(got, ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_lse, ref_lse, atol=3e-2, rtol=3e-2)

    # Small algebraic check plus the real DeepSeek-style 512 + 64 / 512 kernel shape.
    b, hq, n, dk_head, dc, dr, dv_head = 1, 8, 129, 32, 512, 64, 32
    q_content = torch.randn((b, hq, dk_head), device=device, dtype=dtype)
    q_rope = torch.randn((b, hq, dr), device=device, dtype=dtype)
    c_kv = torch.randn((b, n, dc), device=device, dtype=dtype)
    k_rope = torch.randn((b, n, dr), device=device, dtype=dtype)
    w_uk = torch.randn((hq, dk_head, dc), device=device, dtype=dtype) / math.sqrt(dc)
    w_uv = torch.randn((hq, dv_head, dc), device=device, dtype=dtype) / math.sqrt(dc)
    scale = 1.0 / math.sqrt(dk_head + dr)
    unabsorbed = mla_unabsorbed_reference(q_content, q_rope, c_kv, k_rope, w_uk, w_uv, scale=scale)
    q_abs, _, absorbed = mla_absorbed_reference(q_content, q_rope, c_kv, k_rope, w_uk, w_uv, scale=scale)
    torch.testing.assert_close(absorbed, unabsorbed, atol=8e-3, rtol=8e-3)
    latent, _ = mla_decode(
        q_abs.to(dtype),
        q_rope,
        c_kv,
        k_rope,
        torch.tensor([n], device=device, dtype=torch.int32),
        scale=scale,
        num_splits=2,
    )
    kernel_out = torch.einsum("bhc,hvc->bhv", latent.float(), w_uv.float())
    torch.testing.assert_close(kernel_out, absorbed, atol=4e-2, rtol=4e-2)

    print("All KV-cache attention checks passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    run_correctness_checks(dtype)


if __name__ == "__main__":
    main()
