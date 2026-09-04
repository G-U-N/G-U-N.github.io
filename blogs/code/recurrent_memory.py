"""Teaching Triton kernels for recurrent and chunked linear memory.

The code mirrors Part III of the long-context series:

* ``linear_recurrent`` is the state-resident forward schedule used for decode.
* ``linear_chunk`` is the chunked training schedule, with a complete backward.
* ``delta_chunk`` builds the compact-WY rows in Triton and reuses the chunk
  output kernel after constructing DeltaNet's corrected values.

Inputs use the contiguous ``[batch, heads, sequence, width]`` layout.  ``q``
and ``k`` are feature vectors already; choosing phi is deliberately outside
the kernel.  This is educational code, not a replacement for FLA's production
kernels: it omits variable-length packing, gates, grouped heads, and autotuning.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_recurrent_fwd(
    Q,
    K,
    V,
    O,
    H0,
    Z0,
    HT,
    ZT,
    L,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NORMALIZE: tl.constexpr,
    USE_H0: tl.constexpr,
    USE_Z0: tl.constexpr,
    EPS: tl.constexpr,
):
    """One program owns one value tile and keeps its state across time."""
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    offs_k = tl.arange(0, BK)
    offs_v = iv * BV + tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV

    h = tl.zeros([BV, BK], tl.float32)
    if USE_H0:
        h = tl.load(
            H0 + (bh * DV + offs_v[:, None]) * DK + offs_k[None, :],
            mask=mask_v[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

    z = tl.zeros([BK], tl.float32)
    if NORMALIZE and USE_Z0:
        z = tl.load(Z0 + bh * DK + offs_k, mask=mask_k, other=0.0).to(tl.float32)

    for t in range(0, L):
        q = tl.load(Q + (bh * L + t) * DK + offs_k, mask=mask_k, other=0.0)
        k = tl.load(K + (bh * L + t) * DK + offs_k, mask=mask_k, other=0.0)
        v = tl.load(V + (bh * L + t) * DV + offs_v, mask=mask_v, other=0.0)

        # The article uses update-then-read semantics: token t can see itself.
        h += v[:, None] * k[None, :]
        numerator = tl.sum(h * q[None, :], axis=1)
        if NORMALIZE:
            z += k
            denominator = tl.sum(z * q, axis=0) + EPS
            numerator /= denominator

        tl.store(O + (bh * L + t) * DV + offs_v, numerator, mask=mask_v)

    tl.store(
        HT + (bh * DV + offs_v[:, None]) * DK + offs_k[None, :],
        h,
        mask=mask_v[:, None] & mask_k[None, :],
    )
    if NORMALIZE:
        # z is duplicated by the value-tile programs; only one writes it out.
        tl.store(ZT + bh * DK + offs_k, z, mask=(iv == 0) & mask_k)


@triton.jit
def _linear_state_fwd(
    K,
    V,
    H,
    HT,
    L,
    NC,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
):
    """Scan chunks and save the state entering each chunk."""
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    offs_v = iv * BV + tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV
    state = tl.zeros([BV, BK], tl.float32)

    for c in range(0, NC):
        p_h = H + ((bh * NC + c) * DV + offs_v[:, None]) * DK + offs_k[None, :]
        tl.store(p_h, state, mask=mask_v[:, None] & mask_k[None, :])

        rows = c * BT + offs_t
        valid_t = rows < L
        k = tl.load(
            K + (bh * L + rows[:, None]) * DK + offs_k[None, :],
            mask=valid_t[:, None] & mask_k[None, :],
            other=0.0,
        )
        v = tl.load(
            V + (bh * L + rows[:, None]) * DV + offs_v[None, :],
            mask=valid_t[:, None] & mask_v[None, :],
            other=0.0,
        )
        state += tl.dot(tl.trans(v), k)

    tl.store(
        HT + (bh * DV + offs_v[:, None]) * DK + offs_k[None, :],
        state,
        mask=mask_v[:, None] & mask_k[None, :],
    )


@triton.jit
def _linear_z_fwd(
    K,
    Z,
    ZT,
    L,
    NC,
    DK: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """The normalized variant carries one additional feature-mass state."""
    bh = tl.program_id(0)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < DK
    z = tl.zeros([BK], tl.float32)

    for c in range(0, NC):
        tl.store(Z + (bh * NC + c) * DK + offs_k, z, mask=mask_k)
        rows = c * BT + offs_t
        k = tl.load(
            K + (bh * L + rows[:, None]) * DK + offs_k[None, :],
            mask=(rows[:, None] < L) & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        z += tl.sum(k, axis=0)

    tl.store(ZT + bh * DK + offs_k, z, mask=mask_k)


@triton.jit
def _linear_chunk_o_fwd(
    Q,
    K,
    V,
    H,
    Z,
    O,
    DEN,
    L,
    NC,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    NORMALIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    """One program owns one chunk and one output-value tile."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    iv = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    offs_v = iv * BV + tl.arange(0, BV)
    rows = c * BT + offs_t
    valid_t = rows < L
    mask_k = offs_k < DK
    mask_v = offs_v < DV

    q = tl.load(
        Q + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        mask=valid_t[:, None] & mask_k[None, :],
        other=0.0,
    )
    k = tl.load(
        K + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        mask=valid_t[:, None] & mask_k[None, :],
        other=0.0,
    )
    v = tl.load(
        V + (bh * L + rows[:, None]) * DV + offs_v[None, :],
        mask=valid_t[:, None] & mask_v[None, :],
        other=0.0,
    )
    h = tl.load(
        H + ((bh * NC + c) * DV + offs_v[:, None]) * DK + offs_k[None, :],
        mask=mask_v[:, None] & mask_k[None, :],
        other=0.0,
    )

    scores = tl.dot(q, tl.trans(k))
    causal = offs_t[:, None] >= offs_t[None, :]
    visible = valid_t[:, None] & valid_t[None, :] & causal
    scores = tl.where(visible, scores, 0.0)
    numerator = tl.dot(q, tl.trans(h.to(q.dtype)))
    numerator += tl.dot(scores.to(v.dtype), v)

    if NORMALIZE:
        z = tl.load(Z + (bh * NC + c) * DK + offs_k, mask=mask_k, other=0.0)
        denominator = tl.sum(q * z[None, :], axis=1) + tl.sum(scores, axis=1) + EPS
        numerator /= denominator[:, None]
        tl.store(DEN + bh * L + rows, denominator, mask=(iv == 0) & valid_t)

    tl.store(
        O + (bh * L + rows[:, None]) * DV + offs_v[None, :],
        numerator,
        mask=valid_t[:, None] & mask_v[None, :],
    )


@triton.jit
def _linear_bwd_preprocess(
    O,
    DO,
    DEN,
    DN,
    DDEN,
    L,
    DV: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    rows = block * BT + tl.arange(0, BT)
    valid_t = rows < L
    dden = tl.zeros([BT], tl.float32)

    for start_v in range(0, DV, BV):
        offs_v = start_v + tl.arange(0, BV)
        mask = valid_t[:, None] & (offs_v[None, :] < DV)
        o = tl.load(O + (bh * L + rows[:, None]) * DV + offs_v[None, :], mask=mask, other=0.0)
        do = tl.load(DO + (bh * L + rows[:, None]) * DV + offs_v[None, :], mask=mask, other=0.0)
        if NORMALIZE:
            den = tl.load(DEN + bh * L + rows, mask=valid_t, other=1.0)
            dn = do / den[:, None]
            dden -= tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1) / den
        else:
            dn = do
        tl.store(DN + (bh * L + rows[:, None]) * DV + offs_v[None, :], dn, mask=mask)

    if NORMALIZE:
        tl.store(DDEN + bh * L + rows, dden, mask=valid_t)


@triton.jit
def _linear_dh_bwd(
    Q,
    DN,
    DH_NEXT,
    L,
    NC,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
):
    """Reverse-scan the adjoint of the chunk boundary state."""
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    offs_v = iv * BV + tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV
    dh = tl.zeros([BV, BK], tl.float32)

    for step in range(0, NC):
        c = NC - 1 - step
        p_dh = DH_NEXT + ((bh * NC + c) * DV + offs_v[:, None]) * DK + offs_k[None, :]
        tl.store(p_dh, dh, mask=mask_v[:, None] & mask_k[None, :])

        rows = c * BT + offs_t
        valid_t = rows < L
        q = tl.load(
            Q + (bh * L + rows[:, None]) * DK + offs_k[None, :],
            mask=valid_t[:, None] & mask_k[None, :],
            other=0.0,
        )
        dn = tl.load(
            DN + (bh * L + rows[:, None]) * DV + offs_v[None, :],
            mask=valid_t[:, None] & mask_v[None, :],
            other=0.0,
        )
        dh += tl.dot(tl.trans(dn), q)


@triton.jit
def _linear_dz_bwd(
    Q,
    DDEN,
    DZ_NEXT,
    L,
    NC,
    DK: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    bh = tl.program_id(0)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < DK
    dz = tl.zeros([BK], tl.float32)

    for step in range(0, NC):
        c = NC - 1 - step
        tl.store(DZ_NEXT + (bh * NC + c) * DK + offs_k, dz, mask=mask_k)
        rows = c * BT + offs_t
        q = tl.load(
            Q + (bh * L + rows[:, None]) * DK + offs_k[None, :],
            mask=(rows[:, None] < L) & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        dden = tl.load(DDEN + bh * L + rows, mask=rows < L, other=0.0)
        dz += tl.sum(q * dden[:, None], axis=0)


@triton.jit
def _linear_dqdk_bwd(
    Q,
    K,
    V,
    DN,
    DDEN,
    H,
    Z,
    DH_NEXT,
    DZ_NEXT,
    DQ,
    DK_OUT,
    L,
    NC,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    """One chunk owns complete dQ and dK rows; no atomics are needed."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    rows = c * BT + offs_t
    valid_t = rows < L
    mask_k = offs_k < DK

    q = tl.load(
        Q + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        mask=valid_t[:, None] & mask_k[None, :],
        other=0.0,
    )
    k = tl.load(
        K + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        mask=valid_t[:, None] & mask_k[None, :],
        other=0.0,
    )
    dq = tl.zeros([BT, BK], tl.float32)
    dk = tl.zeros([BT, BK], tl.float32)
    dp = tl.zeros([BT, BT], tl.float32)

    for start_v in range(0, DV, BV):
        offs_v = start_v + tl.arange(0, BV)
        mask_v = offs_v < DV
        v = tl.load(
            V + (bh * L + rows[:, None]) * DV + offs_v[None, :],
            mask=valid_t[:, None] & mask_v[None, :],
            other=0.0,
        )
        dn = tl.load(
            DN + (bh * L + rows[:, None]) * DV + offs_v[None, :],
            mask=valid_t[:, None] & mask_v[None, :],
            other=0.0,
        )
        h = tl.load(
            H + ((bh * NC + c) * DV + offs_v[:, None]) * DK + offs_k[None, :],
            mask=mask_v[:, None] & mask_k[None, :],
            other=0.0,
        )
        dh = tl.load(
            DH_NEXT + ((bh * NC + c) * DV + offs_v[:, None]) * DK + offs_k[None, :],
            mask=mask_v[:, None] & mask_k[None, :],
            other=0.0,
        )
        dq += tl.dot(dn, h.to(dn.dtype))
        dp += tl.dot(dn, tl.trans(v))
        dk += tl.dot(v, dh.to(v.dtype))

    causal = offs_t[:, None] >= offs_t[None, :]
    visible = valid_t[:, None] & valid_t[None, :] & causal
    if NORMALIZE:
        dden = tl.load(DDEN + bh * L + rows, mask=valid_t, other=0.0)
        dp += dden[:, None]
        z = tl.load(Z + (bh * NC + c) * DK + offs_k, mask=mask_k, other=0.0)
        dz = tl.load(DZ_NEXT + (bh * NC + c) * DK + offs_k, mask=mask_k, other=0.0)
        dq += dden[:, None] * z[None, :]
        dk += dz[None, :]
    dp = tl.where(visible, dp, 0.0)
    dq += tl.dot(dp.to(k.dtype), k)
    dk += tl.dot(tl.trans(dp.to(q.dtype)), q)

    tl.store(
        DQ + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        dq,
        mask=valid_t[:, None] & mask_k[None, :],
    )
    tl.store(
        DK_OUT + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        dk,
        mask=valid_t[:, None] & mask_k[None, :],
    )


@triton.jit
def _linear_dv_bwd(
    Q,
    K,
    DN,
    DH_NEXT,
    DV_OUT,
    L,
    NC,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
):
    c = tl.program_id(0)
    bh = tl.program_id(1)
    iv = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    offs_v = iv * BV + tl.arange(0, BV)
    rows = c * BT + offs_t
    valid_t = rows < L
    mask_k = offs_k < DK
    mask_v = offs_v < DV

    q = tl.load(
        Q + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        mask=valid_t[:, None] & mask_k[None, :],
        other=0.0,
    )
    k = tl.load(
        K + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        mask=valid_t[:, None] & mask_k[None, :],
        other=0.0,
    )
    dn = tl.load(
        DN + (bh * L + rows[:, None]) * DV + offs_v[None, :],
        mask=valid_t[:, None] & mask_v[None, :],
        other=0.0,
    )
    dh = tl.load(
        DH_NEXT + ((bh * NC + c) * DV + offs_v[:, None]) * DK + offs_k[None, :],
        mask=mask_v[:, None] & mask_k[None, :],
        other=0.0,
    )
    scores = tl.dot(q, tl.trans(k))
    visible = valid_t[:, None] & valid_t[None, :] & (offs_t[:, None] >= offs_t[None, :])
    scores = tl.where(visible, scores, 0.0)
    dv = tl.dot(tl.trans(scores.to(dn.dtype)), dn)
    dv += tl.dot(k, tl.trans(dh.to(k.dtype)))
    tl.store(
        DV_OUT + (bh * L + rows[:, None]) * DV + offs_v[None, :],
        dv,
        mask=valid_t[:, None] & mask_v[None, :],
    )


@triton.jit
def _delta_kkt_fwd(
    K,
    BETA,
    A,
    L,
    NC,
    DK: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Form the one shared strictly lower-triangular beta K K^T matrix."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    rows = c * BT + offs_t
    valid_t = rows < L
    k = tl.load(
        K + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        mask=valid_t[:, None] & (offs_k[None, :] < DK),
        other=0.0,
    )
    beta = tl.load(BETA + bh * L + rows, mask=valid_t, other=0.0).to(tl.float32)
    a = tl.dot(k, tl.trans(k)) * beta[:, None]
    lower = offs_t[:, None] > offs_t[None, :]
    a = tl.where(valid_t[:, None] & valid_t[None, :] & lower, a, 0.0)

    tl.store(
        A
        + ((bh * NC + c) * BT + offs_t[:, None]) * BT
        + offs_t[None, :],
        a,
    )


@triton.jit
def _delta_solve_16_fwd(
    A,
    AINV,
    NC,
    BT: tl.constexpr,
):
    """Invert one 16x16 diagonal block of I + A by forward substitution."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    block = tl.program_id(2)
    offs = tl.arange(0, 16)
    block_offs = block * 16 + offs
    p_a = (
        A
        + ((bh * NC + c) * BT + block_offs[:, None]) * BT
        + block_offs[None, :]
    )
    a = tl.load(p_a).to(tl.float32)
    inverse = tl.zeros([16, 16], tl.float32)

    # A stores only the strict lower triangle. Each new inverse row depends
    # on the rows already completed above it; all 16 columns advance together.
    for r in tl.static_range(0, 16):
        is_r = offs == r
        a_r = tl.sum(tl.where(is_r[:, None], a, 0.0), axis=0)
        inverse_r = is_r.to(tl.float32) - tl.sum(a_r[:, None] * inverse, axis=0)
        inverse = tl.where(is_r[:, None], inverse_r[None, :], inverse)

    tl.store(
        AINV
        + ((bh * NC + c) * BT + block_offs[:, None]) * BT
        + block_offs[None, :],
        inverse,
    )


@triton.jit
def _delta_merge_32_fwd(
    A,
    AINV,
    NC,
    BT: tl.constexpr,
):
    """Merge two inverted 16x16 blocks into one 32x32 inverse."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    pair = tl.program_id(2)
    offs = tl.arange(0, 16)
    base = pair * 32
    top = base + offs
    bottom = base + 16 + offs
    chunk = (bh * NC + c) * BT * BT

    ai11 = tl.load(AINV + chunk + top[:, None] * BT + top[None, :]).to(tl.float32)
    ai22 = tl.load(AINV + chunk + bottom[:, None] * BT + bottom[None, :]).to(tl.float32)
    a21 = tl.load(A + chunk + bottom[:, None] * BT + top[None, :])
    left = tl.dot(ai22, a21, input_precision="ieee")
    ai21 = -tl.dot(left, ai11, input_precision="ieee")
    tl.store(AINV + chunk + bottom[:, None] * BT + top[None, :], ai21)


@triton.jit
def _delta_merge_64_fwd(
    A,
    AINV,
    NC,
    BT: tl.constexpr,
):
    """Merge the two 32x32 diagonal inverses when BT is 64."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    offs = tl.arange(0, 32)
    chunk = (bh * NC + c) * BT * BT
    lower = offs[:, None] >= offs[None, :]

    ai11 = tl.load(AINV + chunk + offs[:, None] * BT + offs[None, :]).to(tl.float32)
    ai11 = tl.where(lower, ai11, 0.0)
    ai22 = tl.load(
        AINV + chunk + (32 + offs[:, None]) * BT + 32 + offs[None, :]
    ).to(tl.float32)
    ai22 = tl.where(lower, ai22, 0.0)
    a21 = tl.load(A + chunk + (32 + offs[:, None]) * BT + offs[None, :])
    left = tl.dot(ai22, a21, input_precision="ieee")
    ai21 = -tl.dot(left, ai11, input_precision="ieee")
    tl.store(
        AINV + chunk + (32 + offs[:, None]) * BT + offs[None, :],
        ai21,
    )


@triton.jit
def _delta_w_fwd(
    AINV,
    K,
    BETA,
    W,
    L,
    NC,
    DK: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Apply the solved triangular system to beta K with one tl.dot."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    rows = c * BT + offs_t
    valid_t = rows < L
    lower = offs_t[:, None] >= offs_t[None, :]
    ainv = tl.load(
        AINV
        + ((bh * NC + c) * BT + offs_t[:, None]) * BT
        + offs_t[None, :]
    )
    ainv = tl.where(lower, ainv, 0.0)
    k = tl.load(
        K + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        mask=valid_t[:, None] & (offs_k[None, :] < DK),
        other=0.0,
    )
    beta = tl.load(BETA + bh * L + rows, mask=valid_t, other=0.0).to(tl.float32)
    rhs = (k * beta[:, None]).to(K.dtype.element_ty)
    w = tl.dot(ainv, rhs)

    tl.store(
        W + (bh * L + rows[:, None]) * DK + offs_k[None, :],
        w,
        mask=valid_t[:, None] & (offs_k[None, :] < DK),
    )


@triton.jit
def _delta_u_fwd(
    AINV,
    V,
    BETA,
    U,
    L,
    NC,
    DV: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
):
    """Apply the same solved triangular system to each beta V tile."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    iv = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_v = iv * BV + tl.arange(0, BV)
    rows = c * BT + offs_t
    valid_t = rows < L
    lower = offs_t[:, None] >= offs_t[None, :]
    ainv = tl.load(
        AINV
        + ((bh * NC + c) * BT + offs_t[:, None]) * BT
        + offs_t[None, :]
    )
    ainv = tl.where(lower, ainv, 0.0)
    v = tl.load(
        V + (bh * L + rows[:, None]) * DV + offs_v[None, :],
        mask=valid_t[:, None] & (offs_v[None, :] < DV),
        other=0.0,
    )
    beta = tl.load(BETA + bh * L + rows, mask=valid_t, other=0.0).to(tl.float32)
    rhs = (v * beta[:, None]).to(V.dtype.element_ty)
    u = tl.dot(ainv, rhs)

    tl.store(
        U + (bh * L + rows[:, None]) * DV + offs_v[None, :],
        u,
        mask=valid_t[:, None] & (offs_v[None, :] < DV),
    )


@triton.jit
def _delta_state_fwd(
    K,
    W,
    U,
    VBAR,
    H,
    L,
    NC,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
):
    """Construct corrected values, then scan the ordinary additive state."""
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    offs_v = iv * BV + tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV
    state = tl.zeros([BV, BK], tl.float32)

    for c in range(0, NC):
        tl.store(
            H + ((bh * NC + c) * DV + offs_v[:, None]) * DK + offs_k[None, :],
            state,
            mask=mask_v[:, None] & mask_k[None, :],
        )
        rows = c * BT + offs_t
        valid_t = rows < L
        k = tl.load(
            K + (bh * L + rows[:, None]) * DK + offs_k[None, :],
            mask=valid_t[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            W + (bh * L + rows[:, None]) * DK + offs_k[None, :],
            mask=valid_t[:, None] & mask_k[None, :],
            other=0.0,
        )
        u = tl.load(
            U + (bh * L + rows[:, None]) * DV + offs_v[None, :],
            mask=valid_t[:, None] & mask_v[None, :],
            other=0.0,
        )
        vbar = u.to(tl.float32) - tl.dot(w, tl.trans(state.to(w.dtype)))
        tl.store(
            VBAR + (bh * L + rows[:, None]) * DV + offs_v[None, :],
            vbar,
            mask=valid_t[:, None] & mask_v[None, :],
        )
        state += tl.dot(tl.trans(vbar.to(k.dtype)), k)


def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int]:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("the Triton kernels require CUDA tensors")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous")
    if q.dtype not in (torch.float16, torch.bfloat16) or not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, and v must share fp16 or bf16 dtype")
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("expected q/k [B,H,L,Df] and v [B,H,L,Dv]")
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    if L < 1:
        raise ValueError("sequence length must be positive")
    if not (1 <= DK <= 128 and 1 <= DV <= 128):
        raise ValueError("this teaching implementation supports Df,Dv <= 128")
    return B, H, L, DK, DV


def _blocks(DK: int, DV: int) -> tuple[int, int]:
    bk = max(16, triton.next_power_of_2(DK))
    bv = min(32, max(16, triton.next_power_of_2(DV)))
    return bk, bv


def linear_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    initial_normalizer: torch.Tensor | None = None,
    normalize: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """State-resident Linear Attention forward for decode or short sequences."""
    B, H, L, DK, DV = _check_qkv(q, k, v)
    BK, BV = _blocks(DK, DV)
    if initial_state is not None and (
        initial_state.shape != (B, H, DV, DK)
        or not initial_state.is_cuda
        or not initial_state.is_contiguous()
    ):
        raise ValueError("initial_state must be contiguous CUDA [B,H,Dv,Df]")
    if normalize and initial_normalizer is not None and (
        initial_normalizer.shape != (B, H, DK)
        or not initial_normalizer.is_cuda
        or not initial_normalizer.is_contiguous()
    ):
        raise ValueError("initial_normalizer must be contiguous CUDA [B,H,Df]")

    o = torch.empty_like(v)
    ht = torch.empty((B, H, DV, DK), device=q.device, dtype=torch.float32)
    zt = torch.empty((B, H, DK), device=q.device, dtype=torch.float32) if normalize else None
    h0 = initial_state if initial_state is not None else q
    z0 = initial_normalizer if initial_normalizer is not None else q
    _linear_recurrent_fwd[(B * H, triton.cdiv(DV, BV))](
        q,
        k,
        v,
        o,
        h0,
        z0,
        ht,
        zt if zt is not None else q,
        L,
        DK=DK,
        DV=DV,
        BK=BK,
        BV=BV,
        NORMALIZE=normalize,
        USE_H0=initial_state is not None,
        USE_Z0=initial_normalizer is not None,
        EPS=eps,
        num_warps=4,
    )
    return o, ht, zt


def _linear_chunk_forward(q, k, v, normalize: bool, eps: float, chunk_size: int):
    B, H, L, DK, DV = _check_qkv(q, k, v)
    if chunk_size not in (16, 32, 64):
        raise ValueError("chunk_size must be 16, 32, or 64")
    NC = triton.cdiv(L, chunk_size)
    BK, BV = _blocks(DK, DV)
    bh = B * H
    nv = triton.cdiv(DV, BV)
    h = torch.empty((B, H, NC, DV, DK), device=q.device, dtype=torch.float32)
    ht = torch.empty((B, H, DV, DK), device=q.device, dtype=torch.float32)
    z = torch.empty((B, H, NC, DK), device=q.device, dtype=torch.float32) if normalize else q.new_empty(1)
    zt = torch.empty((B, H, DK), device=q.device, dtype=torch.float32) if normalize else q.new_empty(1)
    o = torch.empty_like(v)
    den = torch.empty((B, H, L), device=q.device, dtype=torch.float32) if normalize else q.new_empty(1)

    _linear_state_fwd[(bh, nv)](
        k, v, h, ht, L, NC, DK=DK, DV=DV, BK=BK, BV=BV, BT=chunk_size, num_warps=4
    )
    if normalize:
        _linear_z_fwd[(bh,)](k, z, zt, L, NC, DK=DK, BK=BK, BT=chunk_size, num_warps=4)
    _linear_chunk_o_fwd[(NC, bh, nv)](
        q,
        k,
        v,
        h,
        z,
        o,
        den,
        L,
        NC,
        DK=DK,
        DV=DV,
        BK=BK,
        BV=BV,
        BT=chunk_size,
        NORMALIZE=normalize,
        EPS=eps,
        num_warps=4,
    )
    return o, h, z, den


class _LinearChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, normalize: bool, eps: float, chunk_size: int):
        o, h, z, den = _linear_chunk_forward(q, k, v, normalize, eps, chunk_size)
        ctx.save_for_backward(q, k, v, o, h, z, den)
        ctx.normalize = normalize
        ctx.chunk_size = chunk_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, h, z, den = ctx.saved_tensors
        do = do.contiguous()
        B, H, L, DK = q.shape
        DV = v.shape[-1]
        BT = ctx.chunk_size
        NC = triton.cdiv(L, BT)
        BK, BV = _blocks(DK, DV)
        bh = B * H
        nv = triton.cdiv(DV, BV)
        dn = torch.empty_like(v)
        dden = torch.empty((B, H, L), device=q.device, dtype=torch.float32) if ctx.normalize else q.new_empty(1)
        dh_next = torch.empty_like(h)
        dz_next = torch.empty_like(z) if ctx.normalize else q.new_empty(1)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        _linear_bwd_preprocess[(NC, bh)](
            o,
            do,
            den,
            dn,
            dden,
            L,
            DV=DV,
            BV=BV,
            BT=BT,
            NORMALIZE=ctx.normalize,
            num_warps=4,
        )
        _linear_dh_bwd[(bh, nv)](
            q, dn, dh_next, L, NC, DK=DK, DV=DV, BK=BK, BV=BV, BT=BT, num_warps=4
        )
        if ctx.normalize:
            _linear_dz_bwd[(bh,)](
                q, dden, dz_next, L, NC, DK=DK, BK=BK, BT=BT, num_warps=4
            )
        _linear_dqdk_bwd[(NC, bh)](
            q,
            k,
            v,
            dn,
            dden,
            h,
            z,
            dh_next,
            dz_next,
            dq,
            dk,
            L,
            NC,
            DK=DK,
            DV=DV,
            BK=BK,
            BV=BV,
            BT=BT,
            NORMALIZE=ctx.normalize,
            num_warps=4,
        )
        _linear_dv_bwd[(NC, bh, nv)](
            q, k, dn, dh_next, dv, L, NC, DK=DK, DV=DV, BK=BK, BV=BV, BT=BT, num_warps=4
        )
        return dq, dk, dv, None, None, None


def linear_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    normalize: bool = False,
    eps: float = 1e-6,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Chunked training path with a complete q/k/v backward."""
    _check_qkv(q, k, v)
    return _LinearChunk.apply(q, k, v, normalize, eps, chunk_size)


def delta_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Exact chunked DeltaNet forward using a Triton compact-WY preparation."""
    B, H, L, DK, DV = _check_qkv(q, k, v)
    if (
        beta.shape != (B, H, L)
        or not beta.is_cuda
        or not beta.is_contiguous()
        or beta.dtype != q.dtype
    ):
        raise ValueError("beta must be contiguous [B,H,L] with the same CUDA dtype as q")
    if chunk_size not in (16, 32, 64):
        raise ValueError("chunk_size must be 16, 32, or 64")
    if any(x.requires_grad for x in (q, k, v, beta)):
        raise NotImplementedError("the teaching DeltaNet path is forward-only")

    NC = triton.cdiv(L, chunk_size)
    BK, BV = _blocks(DK, DV)
    bh, nv = B * H, triton.cdiv(DV, BV)
    a = torch.empty((B, H, NC, chunk_size, chunk_size), device=q.device, dtype=torch.float32)
    ainv = torch.empty_like(a, dtype=q.dtype)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    vbar = torch.empty_like(v)
    h = torch.empty((B, H, NC, DV, DK), device=q.device, dtype=torch.float32)
    o = torch.empty_like(v)
    dummy = q.new_empty(1)

    _delta_kkt_fwd[(NC, bh)](
        k,
        beta,
        a,
        L,
        NC,
        DK=DK,
        BK=BK,
        BT=chunk_size,
        num_warps=4,
    )
    _delta_solve_16_fwd[(NC, bh, chunk_size // 16)](
        a,
        ainv,
        NC,
        BT=chunk_size,
        num_warps=4,
    )
    if chunk_size >= 32:
        _delta_merge_32_fwd[(NC, bh, chunk_size // 32)](
            a,
            ainv,
            NC,
            BT=chunk_size,
            num_warps=4,
        )
    if chunk_size == 64:
        _delta_merge_64_fwd[(NC, bh)](
            a,
            ainv,
            NC,
            BT=chunk_size,
            num_warps=4,
        )
    _delta_w_fwd[(NC, bh)](
        ainv,
        k,
        beta,
        w,
        L,
        NC,
        DK=DK,
        BK=BK,
        BT=chunk_size,
        num_warps=4,
    )
    _delta_u_fwd[(NC, bh, nv)](
        ainv,
        v,
        beta,
        u,
        L,
        NC,
        DV=DV,
        BV=BV,
        BT=chunk_size,
        num_warps=4,
    )
    _delta_state_fwd[(bh, nv)](
        k,
        w,
        u,
        vbar,
        h,
        L,
        NC,
        DK=DK,
        DV=DV,
        BK=BK,
        BV=BV,
        BT=chunk_size,
        num_warps=4,
    )
    _linear_chunk_o_fwd[(NC, bh, nv)](
        q,
        k,
        vbar,
        h,
        dummy,
        o,
        dummy,
        L,
        NC,
        DK=DK,
        DV=DV,
        BK=BK,
        BV=BV,
        BT=chunk_size,
        NORMALIZE=False,
        EPS=0.0,
        num_warps=4,
    )
    return o


def linear_reference(q, k, v, *, normalize=False, eps=1e-6):
    """Readable token recurrence used for forward and gradient checks."""
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    state = torch.zeros((B, H, DV, DK), device=q.device, dtype=torch.float32)
    z = torch.zeros((B, H, DK), device=q.device, dtype=torch.float32)
    outputs = []
    for t in range(L):
        kt = k[:, :, t].float()
        vt = v[:, :, t].float()
        qt = q[:, :, t].float()
        state = state + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        numerator = torch.einsum("bhvk,bhk->bhv", state, qt)
        if normalize:
            z = z + kt
            numerator = numerator / (torch.einsum("bhk,bhk->bh", z, qt).unsqueeze(-1) + eps)
        outputs.append(numerator)
    return torch.stack(outputs, dim=2).to(v.dtype)


def delta_reference(q, k, v, beta):
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    state = torch.zeros((B, H, DV, DK), device=q.device, dtype=torch.float32)
    outputs = []
    for t in range(L):
        kt = k[:, :, t].float()
        vt = v[:, :, t].float()
        prediction = torch.einsum("bhvk,bhk->bhv", state, kt)
        corrected = beta[:, :, t].float().unsqueeze(-1) * (vt - prediction)
        state = state + corrected.unsqueeze(-1) * kt.unsqueeze(-2)
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, :, t].float()))
    return torch.stack(outputs, dim=2).to(v.dtype)


def _self_test():
    if not torch.cuda.is_available():
        raise RuntimeError("the Triton self-test requires an NVIDIA CUDA GPU")
    torch.manual_seed(0)
    B, H, L, DK, DV = 1, 2, 45, 32, 24
    dtype = torch.bfloat16
    for normalize in (False, True):
        q = torch.randn((B, H, L, DK), device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        if normalize:
            q, k = torch.nn.functional.softplus(q), torch.nn.functional.softplus(k)
        v = torch.randn((B, H, L, DV), device="cuda", dtype=dtype)

        recurrent, _, _ = linear_recurrent(q, k, v, normalize=normalize)
        reference = linear_reference(q, k, v, normalize=normalize)
        torch.testing.assert_close(recurrent, reference, atol=6e-2, rtol=6e-2)
        for chunk_size in (16, 64):  # multiple chunks, then L < chunk_size
            chunked = linear_chunk(q, k, v, normalize=normalize, chunk_size=chunk_size)
            torch.testing.assert_close(chunked, reference, atol=6e-2, rtol=6e-2)

        do = torch.randn_like(v)
        q1, k1, v1 = (x.detach().requires_grad_(True) for x in (q, k, v))
        q2, k2, v2 = (x.detach().requires_grad_(True) for x in (q, k, v))
        grads_tri = torch.autograd.grad(linear_chunk(q1, k1, v1, normalize=normalize, chunk_size=16), (q1, k1, v1), do)
        grads_ref = torch.autograd.grad(linear_reference(q2, k2, v2, normalize=normalize), (q2, k2, v2), do)
        for actual, expected in zip(grads_tri, grads_ref):
            torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

    q = torch.randn((B, H, L, DK), device="cuda", dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn((B, H, L, DV), device="cuda", dtype=dtype)
    beta = torch.sigmoid(torch.randn((B, H, L), device="cuda", dtype=dtype))
    with torch.no_grad():
        reference = delta_reference(q, k, v, beta)
        for chunk_size in (16, 32, 64):
            torch.testing.assert_close(
                delta_chunk(q, k, v, beta, chunk_size=chunk_size),
                reference,
                atol=8e-2,
                rtol=8e-2,
            )
    print("Recurrent, chunk forward/backward, and DeltaNet forward match PyTorch references.")


if __name__ == "__main__":
    _self_test()
