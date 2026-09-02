"""Minimal FlashAttention forward/backward written in Triton.

This is a teaching implementation.  It follows the two-traversal ownership
pattern used by Triton's fused-attention tutorial, while using natural
exponentials so that the saved LSE is exactly torch.logsumexp(scores).

Supported layout: contiguous [batch, heads, sequence, head_dim] tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd(
    Q,
    K,
    V,
    O,
    LSE,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    base = b * stride_b + h * stride_h

    rows = q_block * BLOCK + tl.arange(0, BLOCK)
    cols_in_tile = tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    q = tl.load(
        Q + base + rows[:, None] * D + dims[None, :],
        mask=rows[:, None] < L,
        other=0.0,
    )

    # One online-softmax state per query row.
    m = tl.full([BLOCK], -float("inf"), tl.float32)
    ell = tl.zeros([BLOCK], tl.float32)
    acc = tl.zeros([BLOCK, D], tl.float32)

    for start_n in range(0, L, BLOCK):
        cols = start_n + cols_in_tile
        k = tl.load(
            K + base + cols[:, None] * D + dims[None, :],
            mask=cols[:, None] < L,
            other=0.0,
        )
        v = tl.load(
            V + base + cols[:, None] * D + dims[None, :],
            mask=cols[:, None] < L,
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * SCALE
        visible = (rows[:, None] < L) & (cols[None, :] < L)
        if CAUSAL:
            visible = visible & (rows[:, None] >= cols[None, :])
        scores = tl.where(visible, scores, -1.0e6)

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(visible, p, 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        ell = ell * alpha + tl.sum(p, axis=1)
        m = m_new

    out = acc / ell[:, None]
    tl.store(
        O + base + rows[:, None] * D + dims[None, :],
        out,
        mask=rows[:, None] < L,
    )
    tl.store(LSE + bh * L + rows, m + tl.log(ell), mask=rows < L)


@triton.jit
def _flash_bwd_preprocess(
    O,
    DO,
    Delta,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    base = b * stride_b + h * stride_h
    rows = block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)

    o = tl.load(
        O + base + rows[:, None] * D + dims[None, :],
        mask=rows[:, None] < L,
        other=0.0,
    )
    do = tl.load(
        DO + base + rows[:, None] * D + dims[None, :],
        mask=rows[:, None] < L,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + bh * L + rows, delta, mask=rows < L)


@triton.jit
def _flash_bwd(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    Delta,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    base = b * stride_b + h * stride_h
    in_tile = tl.arange(0, BLOCK)
    dims = tl.arange(0, D)

    # Traversal 1: this program owns one KV tile and finishes dK and dV.
    cols = tile * BLOCK + in_tile
    k = tl.load(
        K + base + cols[:, None] * D + dims[None, :],
        mask=cols[:, None] < L,
        other=0.0,
    )
    v = tl.load(
        V + base + cols[:, None] * D + dims[None, :],
        mask=cols[:, None] < L,
        other=0.0,
    )
    dk = tl.zeros([BLOCK, D], tl.float32)
    dv = tl.zeros([BLOCK, D], tl.float32)

    for start_m in range(0, L, BLOCK):
        rows = start_m + in_tile
        q = tl.load(
            Q + base + rows[:, None] * D + dims[None, :],
            mask=rows[:, None] < L,
            other=0.0,
        )
        do = tl.load(
            DO + base + rows[:, None] * D + dims[None, :],
            mask=rows[:, None] < L,
            other=0.0,
        )
        lse = tl.load(LSE + bh * L + rows, mask=rows < L, other=0.0)
        delta = tl.load(Delta + bh * L + rows, mask=rows < L, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * SCALE
        visible = (rows[:, None] < L) & (cols[None, :] < L)
        if CAUSAL:
            visible = visible & (rows[:, None] >= cols[None, :])
        scores = tl.where(visible, scores, -1.0e6)
        p = tl.exp(scores - lse[:, None])
        p = tl.where(visible, p, 0.0)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None])

        dv += tl.dot(tl.trans(p.to(q.dtype)), do)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * SCALE

    tl.store(
        DK + base + cols[:, None] * D + dims[None, :],
        dk,
        mask=cols[:, None] < L,
    )
    tl.store(
        DV + base + cols[:, None] * D + dims[None, :],
        dv,
        mask=cols[:, None] < L,
    )

    # Traversal 2: the same program id now owns one query tile and finishes dQ.
    rows = tile * BLOCK + in_tile
    q = tl.load(
        Q + base + rows[:, None] * D + dims[None, :],
        mask=rows[:, None] < L,
        other=0.0,
    )
    do = tl.load(
        DO + base + rows[:, None] * D + dims[None, :],
        mask=rows[:, None] < L,
        other=0.0,
    )
    lse = tl.load(LSE + bh * L + rows, mask=rows < L, other=0.0)
    delta = tl.load(Delta + bh * L + rows, mask=rows < L, other=0.0)
    dq = tl.zeros([BLOCK, D], tl.float32)

    for start_n in range(0, L, BLOCK):
        cols = start_n + in_tile
        k = tl.load(
            K + base + cols[:, None] * D + dims[None, :],
            mask=cols[:, None] < L,
            other=0.0,
        )
        v = tl.load(
            V + base + cols[:, None] * D + dims[None, :],
            mask=cols[:, None] < L,
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * SCALE
        visible = (rows[:, None] < L) & (cols[None, :] < L)
        if CAUSAL:
            visible = visible & (rows[:, None] >= cols[None, :])
        scores = tl.where(visible, scores, -1.0e6)
        p = tl.exp(scores - lse[:, None])
        p = tl.where(visible, p, 0.0)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(q.dtype), k) * SCALE

    tl.store(
        DQ + base + rows[:, None] * D + dims[None, :],
        dq,
        mask=rows[:, None] < L,
    )


def _check_inputs(q, k, v):
    assert q.is_cuda and q.is_contiguous()
    assert q.shape == k.shape == v.shape
    assert k.is_contiguous() and v.is_contiguous()
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.shape[-1] in (64, 128)


def flash_forward(q, k, v, causal=False):
    _check_inputs(q, k, v)
    B, H, L, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    block = 64
    grid = (triton.cdiv(L, block), B * H)
    _flash_fwd[grid](
        q,
        k,
        v,
        o,
        lse,
        q.stride(0),
        q.stride(1),
        H=H,
        L=L,
        D=D,
        SCALE=D**-0.5,
        CAUSAL=causal,
        BLOCK=block,
        num_warps=4,
        num_stages=2,
    )
    return o, lse


def flash_backward(q, k, v, o, lse, do, causal=False):
    _check_inputs(q, k, v)
    assert o.shape == do.shape == q.shape
    assert o.is_contiguous() and do.is_contiguous()
    B, H, L, D = q.shape
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    delta = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    block = 64
    grid = (triton.cdiv(L, block), B * H)

    _flash_bwd_preprocess[grid](
        o,
        do,
        delta,
        q.stride(0),
        q.stride(1),
        H=H,
        L=L,
        D=D,
        BLOCK=block,
        num_warps=4,
    )
    _flash_bwd[grid](
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        lse,
        delta,
        q.stride(0),
        q.stride(1),
        H=H,
        L=L,
        D=D,
        SCALE=D**-0.5,
        CAUSAL=causal,
        BLOCK=block,
        num_warps=4,
        num_stages=2,
    )
    return dq, dk, dv


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False):
        o, lse = flash_forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = flash_backward(q, k, v, o, lse, do.contiguous(), ctx.causal)
        return dq, dk, dv, None


flash_attention = _FlashAttention.apply


def _reference(q, k, v, causal):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        mask = torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, -float("inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype), scores


def check_against_torch(causal, dtype=torch.float16):
    torch.manual_seed(0)
    shape = (1, 2, 128, 64)
    q0 = torch.randn(shape, device="cuda", dtype=dtype) * 0.5
    k0 = torch.randn(shape, device="cuda", dtype=dtype) * 0.5
    v0 = torch.randn(shape, device="cuda", dtype=dtype) * 0.5
    do = torch.randn_like(q0)

    q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_(True) for x in (q0, k0, v0)]
    o_ref, scores_ref = _reference(q_ref, k_ref, v_ref, causal)
    grads_ref = torch.autograd.grad(o_ref, (q_ref, k_ref, v_ref), do)

    q_tri, k_tri, v_tri = [x.detach().clone().requires_grad_(True) for x in (q0, k0, v0)]
    o_tri = flash_attention(q_tri, k_tri, v_tri, causal)
    grads_tri = torch.autograd.grad(o_tri, (q_tri, k_tri, v_tri), do)
    _, lse_tri = flash_forward(q0, k0, v0, causal)
    lse_ref = torch.logsumexp(scores_ref, dim=-1)

    torch.testing.assert_close(o_tri, o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse_tri, lse_ref, atol=3e-2, rtol=3e-2)
    for actual, expected in zip(grads_tri, grads_ref):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This example requires a CUDA GPU."
    check_against_torch(causal=False)
    check_against_torch(causal=True)
    if torch.cuda.is_bf16_supported():
        check_against_torch(causal=False, dtype=torch.bfloat16)
        check_against_torch(causal=True, dtype=torch.bfloat16)
    print("Triton forward, LSE reconstruction, and backward match PyTorch.")
