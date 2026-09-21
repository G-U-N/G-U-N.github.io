#!/usr/bin/env python3
"""A correctness-oriented single-node Expert Parallel MoE reference.

The ``torch-triton`` backend intentionally exposes the data movement instead of
hiding it in a framework.  Every rank owns a contiguous shard of experts and
uses explicit ``ProcessGroup`` objects plus ``all_to_all_single`` for dispatch
and combine. AllToAllRows reverses communication in autograd, and GroupedLinear
uses Triton grouped GEMMs for both input and weight gradients. Calling
loss.backward() differentiates the experts, selected softmax, and router
projection, as in the article. A separate PyTorch reference checks the result.
With --check, outputs and gradients are compared against PyTorch autograd.
This runnable example uses one EP group covering the entire torchrun world;
--ep-size must equal world_size. It has no TP, PP, CP, or Expert-DP replicas.
The checked objective sums the local losses across the EP world; router
weight gradients are summed across its replicas, while expert gradients stay
on their owners. Multiple EP groups and Expert-DP synchronization are
explained in the article.

The optional ``deepseek`` backend calls the V2 APIs from the official
DeepEP/DeepGEMM repositories.  Those libraries are Hopper/Blackwell-oriented
and are not silently emulated: when an API or hardware requirement is missing,
the program reports the exact prerequisite and exits.

Examples (one node):

    torchrun --standalone --nproc-per-node=4 blogs/code/moe_ep_hopper.py \
        --backend torch-triton --phase train --check

    torchrun --standalone --nproc-per-node=8 blogs/code/moe_ep_hopper.py \
        --backend deepseek --phase decode --check

    torchrun --standalone --nproc-per-node=4 blogs/code/moe_ep_hopper.py \
        --backend both --phase decode --check --bench-iters 30

Timing excludes input generation and first-use compilation/library setup.
It includes router score/weight computation, dynamic packing, CPU count
synchronization, expert compute, and combine. Top-k indices are fixed across
runs so both backends and the gradient reference use identical routes.
Timing compares whole reference pipelines.

The DeepSeek path follows the public V2 interfaces documented at:
https://github.com/deepseek-ai/DeepEP and
https://github.com/deepseek-ai/DeepGEMM.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
import triton
import triton.language as tl


def make_contiguous_schedule(counts, block_m, *, device):
    expert_tiles, row_tiles = [], []
    for expert, count in enumerate(counts.tolist()):
        for row_tile in range((count + block_m - 1) // block_m):
            expert_tiles.append(expert)
            row_tiles.append(row_tile)
    return (
        torch.tensor(expert_tiles, device=device, dtype=torch.int32),
        torch.tensor(row_tiles, device=device, dtype=torch.int32),
    )


@triton.jit
def _m_grouped_nt_contiguous(
    A, B, C, offsets,
    tile_expert, tile_m,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(tile_expert + pid_m)
    local_tile_m = tl.load(tile_m + pid_m)

    start = tl.load(offsets + expert)
    end = tl.load(offsets + expert + 1)
    rows = local_tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # B is [E, N, K]. Pointer arithmetic exposes B[expert].T as [K, N].
    for start_k in range(0, K, BLOCK_K):
        ks = start_k + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + (start + rows)[:, None] * stride_am
              + ks[None, :] * stride_ak,
            mask=(start + rows[:, None] < end) & (ks[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + expert * stride_be
              + ks[:, None] * stride_bk
              + cols[None, :] * stride_bn,
            mask=(ks[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        C + (start + rows)[:, None] * stride_cm
          + cols[None, :] * stride_cn,
        acc.to(C.dtype.element_ty),
        mask=(start + rows[:, None] < end) & (cols[None, :] < N),
    )


def grouped_mm_contiguous(
    a, b, offsets, *, block_m=32, block_n=64, block_k=32
):
    # a: [sum_e M_e, K], b: [E, N, K], offsets: [E + 1]
    assert a.ndim == 2 and b.ndim == 3
    assert b.shape[2] == a.shape[1]
    assert a.dtype == b.dtype
    a, b = a.contiguous(), b.contiguous()

    counts = offsets[1:] - offsets[:-1]
    tile_expert, tile_m = make_contiguous_schedule(
        counts.cpu(), block_m, device=a.device
    )
    out = torch.empty(
        (a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype
    )
    if tile_expert.numel() == 0:
        return out

    grid = (
        tile_expert.numel(),
        triton.cdiv(b.shape[1], block_n),
    )
    _m_grouped_nt_contiguous[grid](
        a, b, out, offsets, tile_expert, tile_m,
        b.shape[1], a.shape[1],
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(2), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


@triton.jit
def _grouped_weight_grad_kernel(
    A, dC, dB, offsets,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_M: tl.constexpr,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    stride_be, stride_bk, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_k = tl.cdiv(K, BLOCK_K)
    tiles_n = tl.cdiv(N, BLOCK_N)
    expert = pid // (tiles_k * tiles_n)
    rem = pid % (tiles_k * tiles_n)
    tile_k = rem // tiles_n
    tile_n = rem % tiles_n

    start = tl.load(offsets + expert)
    end = tl.load(offsets + expert + 1)
    ks = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    ns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_K, BLOCK_N), tl.float32)

    # acc = A_e.T @ dC_e; only this expert's M_e rows are valid.
    for row_start in range(0, MAX_M, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        a = tl.load(
            A + (start + rows)[:, None] * stride_am
              + ks[None, :] * stride_ak,
            mask=(start + rows[:, None] < end) & (ks[None, :] < K),
            other=0.0,
        )
        dc = tl.load(
            dC + (start + rows)[:, None] * stride_cm
               + ns[None, :] * stride_cn,
            mask=(start + rows[:, None] < end) & (ns[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(tl.trans(a), dc)

    # Store dB as [E, N, K], the same NT layout used in forward.
    tl.store(
        dB + expert * stride_be
           + ns[:, None] * stride_bn
           + ks[None, :] * stride_bk,
        tl.trans(acc).to(dB.dtype.element_ty),
        mask=(ns[:, None] < N) & (ks[None, :] < K),
    )


def grouped_weight_grad(
    a, dc, offsets, *, block_m=32, block_n=64, block_k=32
):
    a, dc = a.contiguous(), dc.contiguous()
    E = offsets.numel() - 1
    K, N = a.shape[1], dc.shape[1]
    db = torch.empty((E, N, K), device=a.device, dtype=a.dtype)
    max_m = max(1, int((offsets[1:] - offsets[:-1]).max().item()))
    grid = (
        E * math.ceil(K / block_k) * math.ceil(N / block_n),
    )
    _grouped_weight_grad_kernel[grid](
        a, dc, db, offsets, N, K, max_m,
        a.stride(0), a.stride(1),
        dc.stride(0), dc.stride(1),
        db.stride(0), db.stride(2), db.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return db


def grouped_mm_backward(a, b, dc, offsets):
    # Forward is C_e = A_e @ B_e.T with B=[E,N,K].
    da = grouped_mm_contiguous(
        dc, b.transpose(1, 2).contiguous(), offsets
    )
    db = grouped_weight_grad(a, dc, offsets)
    return da, db


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This script targets a single-node CUDA process group. "
            "Use H20/H100 GPUs with torchrun; CPU/Gloo is intentionally not a fallback."
        )


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    raise RuntimeError(
        "FP8 is guarded in this reference. Use --dtype bf16 for the runnable "
        "Triton path; the official DeepGEMM FP8 path additionally needs its "
        "fine-grained scale-factor tensors and matching layout."
    )


def _init_distributed() -> tuple[int, int, int, dist.ProcessGroup]:
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    # Keep group creation explicit even when the EP group is the full world.
    ep_group = dist.new_group(ranks=list(range(world)), backend="nccl")
    return rank, world, local_rank, ep_group


def exchange_counts(send_counts, ep_group):
    ep_size = dist.get_world_size(ep_group)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(
        recv_counts,
        send_counts,
        output_split_sizes=[1] * ep_size,
        input_split_sizes=[1] * ep_size,
        group=ep_group,
    )
    return recv_counts


def all_to_all_rows(send, output_splits, input_splits, ep_group):
    recv = torch.empty(
        (sum(output_splits), *send.shape[1:]),
        device=send.device,
        dtype=send.dtype,
    )
    dist.all_to_all_single(
        recv,
        send.contiguous(),
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=ep_group,
    )
    return recv


class AllToAllRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, send, output_splits, input_splits, group):
        ctx.output_splits = list(output_splits)
        ctx.input_splits = list(input_splits)
        ctx.group = group
        return all_to_all_rows(
            send, ctx.output_splits, ctx.input_splits, group
        )

    @staticmethod
    def backward(ctx, grad_recv):
        # Reverse the edges used by forward.
        grad_send = all_to_all_rows(
            grad_recv.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            ctx.group,
        )
        return grad_send, None, None, None


def ep_dispatch(x, topk_idx, num_experts, ep_group):
    # x: [T,d], topk_idx: [T,k]
    T, top_k = topk_idx.shape
    ep_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    assert num_experts % ep_size == 0
    experts_per_rank = num_experts // ep_size

    # 1. Expand routes and pack by destination.
    token = torch.arange(T, device=x.device, dtype=torch.int64)
    token = token[:, None].expand(T, top_k).reshape(-1)
    slot = torch.arange(top_k, device=x.device, dtype=torch.int64)
    slot = slot[None, :].expand(T, top_k).reshape(-1)
    expert = topk_idx.reshape(-1).to(torch.int64)

    owner = expert // experts_per_rank
    local_expert = expert % experts_per_rank

    # Primary key: destination group rank. Secondary key: its local expert.
    route_order = torch.argsort(
        owner * experts_per_rank + local_expert,
        stable=True,
    )
    send_x = x[token[route_order]]
    send_expert = local_expert[route_order].to(torch.int32)
    send_meta = torch.stack([
        torch.full_like(token, ep_rank),  # source group rank
        token,                           # source token
        slot,                            # source top-k slot
    ], dim=1)[route_order]

    # 2. Exchange counts and prepare split lists.
    send_counts = torch.bincount(owner, minlength=ep_size).to(torch.int64)
    recv_counts = exchange_counts(send_counts, ep_group)
    send_splits = send_counts.cpu().tolist()
    recv_splits = recv_counts.cpu().tolist()

    # 3. Exchange token rows and their metadata.
    recv_x = AllToAllRows.apply(
        send_x, recv_splits, send_splits, ep_group
    )
    recv_expert = all_to_all_rows(
        send_expert, recv_splits, send_splits, ep_group
    )
    recv_meta = all_to_all_rows(
        send_meta, recv_splits, send_splits, ep_group
    )

    # 4. all-to-all returns source-major segments. Grouped GEMM needs
    # rows from the same local expert to be contiguous.
    expert_order = torch.argsort(recv_expert, stable=True)
    recv_x = recv_x[expert_order].contiguous()
    recv_expert = recv_expert[expert_order].contiguous()
    recv_meta = recv_meta[expert_order].contiguous()

    counts = torch.bincount(
        recv_expert.to(torch.int64), minlength=experts_per_rank
    ).to(torch.int32)
    offsets = torch.cat([
        torch.zeros(1, device=x.device, dtype=torch.int32),
        counts.cumsum(0),
    ])
    context = {
        "send_splits": send_splits,
        "recv_splits": recv_splits,
        "route_order": route_order,
        "expert_order": expert_order,
        "recv_meta": recv_meta,
    }
    return recv_x, recv_expert, offsets, context


class GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offsets):
        ctx.save_for_backward(a, b, offsets)
        return grouped_mm_contiguous(a, b, offsets)

    @staticmethod
    def backward(ctx, dc):
        a, b, offsets = ctx.saved_tensors
        da, db = grouped_mm_backward(a, b, dc.contiguous(), offsets)
        return da, db, None


def grouped_linear(a, b, offsets):
    return GroupedLinear.apply(a, b, offsets)


def local_expert_swiglu(
    recv_x, offsets, w_gate, w_up, w_down
):
    gate = grouped_linear(recv_x, w_gate, offsets)
    up = grouped_linear(recv_x, w_up, offsets)
    hidden = torch.nn.functional.silu(gate) * up
    return grouped_linear(hidden, w_down, offsets)


def ep_combine(
    packed_y, topk_weight, dispatch_context, T, ep_group
):
    meta = dispatch_context["recv_meta"]

    # One contiguous segment for every source group rank.
    return_order = torch.argsort(meta[:, 0], stable=True)
    send_y = packed_y[return_order]
    send_meta = meta[return_order]

    # Forward dispatch received recv_splits rows from each source.
    # Combine sends those rows back and receives send_splits rows.
    input_splits = dispatch_context["recv_splits"]
    output_splits = dispatch_context["send_splits"]
    returned_y = AllToAllRows.apply(
        send_y, output_splits, input_splits, ep_group
    )
    returned_meta = all_to_all_rows(
        send_meta, output_splits, input_splits, ep_group
    )

    output = torch.zeros(
        (T, packed_y.shape[1]),
        device=packed_y.device,
        dtype=torch.float32,
    )
    source_token = returned_meta[:, 1].to(torch.int64)
    source_slot = returned_meta[:, 2].to(torch.int64)
    route_weight = topk_weight[source_token, source_slot]
    output.index_add_(
        0,
        source_token,
        returned_y.float() * route_weight[:, None],
    )
    return output.to(packed_y.dtype), (returned_y, returned_meta)


def ep_moe_forward(
    x, topk_idx, topk_weight,
    w_gate, w_up, w_down,
    num_experts, ep_group,
):
    recv_x, _, offsets, context = ep_dispatch(
        x, topk_idx, num_experts, ep_group
    )
    packed_y = local_expert_swiglu(
        recv_x, offsets, w_gate, w_up, w_down
    )
    output, combine_context = ep_combine(
        packed_y, topk_weight, context, x.shape[0], ep_group
    )
    return output, (context, combine_context)


def _make_inputs(args: argparse.Namespace, rank: int, world: int, device: torch.device):
    torch.manual_seed(args.seed + rank)
    x = torch.randn((args.tokens, args.hidden), device=device, dtype=torch.bfloat16) / 8
    # Replicated router parameters, but different local tokens on each rank.
    torch.manual_seed(args.seed + 500)
    w_router = torch.randn((args.num_experts, args.hidden), device=device) / args.hidden**0.5
    logits = x.float() @ w_router.T
    topk_idx = logits.topk(args.top_k, dim=-1).indices
    topk_weight = logits.gather(1, topk_idx).softmax(-1)
    local_e = args.num_experts // world
    torch.manual_seed(args.seed + 1000 + rank)
    w_gate = torch.randn((local_e, args.ffn, args.hidden), device=device, dtype=torch.bfloat16) / (args.hidden**0.5)
    w_up = torch.randn_like(w_gate) / (args.hidden**0.5)
    w_down = torch.randn((local_e, args.hidden, args.ffn), device=device, dtype=torch.bfloat16) / (args.ffn**0.5)
    return x, topk_idx.to(torch.int32), topk_weight, (w_gate, w_up, w_down), w_router


def run_torch_triton(args: argparse.Namespace, rank: int, world: int, group: dist.ProcessGroup, inputs=None) -> dict:
    """Run the article's differentiable EP path and use loss.backward()."""
    if args.dtype != "bf16":
        raise RuntimeError("torch-triton backend currently supports only --dtype bf16")
    device = torch.device("cuda", torch.cuda.current_device())
    base_x, topk_idx, _, base_weights, base_router = inputs or _make_inputs(args, rank, world, device)
    train = args.phase == "train"
    # Fresh leaves prevent benchmark iterations from accumulating gradients.
    x = base_x.detach().requires_grad_(train)
    w_router = base_router.detach().requires_grad_(train)
    weights = tuple(w.detach().requires_grad_(train) for w in base_weights)
    with torch.set_grad_enabled(train):
        logits = x.float() @ w_router.T
        topk_weight = logits.gather(1, topk_idx.long()).softmax(-1)
        if train:
            topk_weight.retain_grad()
        output, _ = ep_moe_forward(
            x, topk_idx, topk_weight, *weights, args.num_experts, group
        )
        result = {
            "output": output, "x": x, "topk_idx": topk_idx,
            "topk_weight": topk_weight, "weights": weights, "w_router": w_router,
        }
        if train:
            torch.manual_seed(args.seed + 2000 + rank)
            dout = torch.randn_like(output) / args.hidden**0.5
            loss = (output.float() * dout.float()).sum()
            loss.backward()
            # This example has EP=world and no Expert-DP replicas. Only the
            # replicated router needs a parameter-gradient collective.
            # SUM matches the sum of the ranks' local losses used by the oracle.
            # No DDP/FSDP communication hooks are installed in this script.
            dist.all_reduce(w_router.grad, op=dist.ReduceOp.SUM, group=group)
            result.update(
                loss=loss.detach(), dout=dout, dx=x.grad,
                dweight=topk_weight.grad, dw_router=w_router.grad,
                dw_gate=weights[0].grad, dw_up=weights[1].grad,
                dw_down=weights[2].grad,
            )
    return result


def _deepseek_error(exc: BaseException) -> RuntimeError:
    return RuntimeError(
        "DeepSeek backend unavailable. Install the official DeepEP V2 and "
        "DeepGEMM revisions with blogs/code/setup_moe_hopper.sh, then run on "
        "SM90 (CUDA Toolkit >= 12.3) or SM100 (Toolkit >= 12.9), "
        "C++20, PyTorch >= 2.10, NCCL "
        ">= 2.30.4, and NVLink. Original error: " + repr(exc)
    )


def _check_deepseek_environment(deep_gemm):
    """Fail before launching when the pinned libraries' prerequisites differ."""
    import re
    import shutil
    import subprocess
    from pathlib import Path
    from packaging.version import Version
    from torch.utils.cpp_extension import CUDA_HOME

    major, minor = torch.cuda.get_device_capability()
    if major not in (9, 10):
        raise RuntimeError(f"this backend targets SM90/SM100; found SM{major}{minor}")
    if Version(torch.__version__.split("+")[0]) < Version("2.10"):
        raise RuntimeError(f"DeepEP V2 needs PyTorch >= 2.10; found {torch.__version__}")
    nvcc = os.environ.get("DG_JIT_NVCC_COMPILER")
    if not nvcc:
        nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME else shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("CUDA Toolkit nvcc is required for JIT compilation")
    version_text = subprocess.check_output([nvcc, "--version"], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"release\s+(\d+)\.(\d+)", version_text)
    required = (12, 9) if major == 10 else (12, 3)
    if match is None or tuple(map(int, match.groups())) < required:
        raise RuntimeError(f"SM{major}0 needs CUDA Toolkit >= {required[0]}.{required[1]}")
    if not dist.is_nccl_available():
        raise RuntimeError("PyTorch must include the NCCL backend")
    version = torch.cuda.nccl.version()
    if not isinstance(version, tuple):
        version = (version // 10000, (version // 100) % 100, version % 100)
    if version < (2, 30, 4):
        raise RuntimeError(f"DeepEP V2 needs NCCL >= 2.30.4; found {version}")
    for name in ("m_grouped_bf16_gemm_nt_contiguous", "get_mk_alignment_for_contiguous_layout"):
        if not callable(getattr(deep_gemm, name, None)):
            raise RuntimeError(f"installed DeepGEMM lacks {name}; use the pinned revision")


def run_deepseek(args: argparse.Namespace, rank: int, world: int, group: dist.ProcessGroup, inputs=None) -> dict:
    if args.dtype != "bf16":
        raise RuntimeError(
            "The DeepSeek FP8 API needs explicit per-group scale-factor layouts. "
            "This reference intentionally does not invent those tensors; use "
            "--dtype bf16 or the official DeepGEMM FP8 tests."
        )
    try:
        import deep_gemm
        import deep_ep
        from deep_ep import ElasticBuffer
    except Exception as exc:  # pragma: no cover - depends on target machine
        raise _deepseek_error(exc) from exc
    if not getattr(args, "_deepseek_ready", False):
        try:
            _check_deepseek_environment(deep_gemm)
            alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            if args.expert_alignment != alignment:
                raise RuntimeError(
                    f"DeepEP expert alignment ({args.expert_alignment}) must match "
                    f"DeepGEMM's layout alignment ({alignment})"
                )
        except Exception as exc:
            raise _deepseek_error(exc) from exc
        args._deepseek_ready = True
    device = torch.device("cuda", torch.cuda.current_device())
    x, topk_idx, _, weights, w_router = inputs or _make_inputs(args, rank, world, device)
    # Match the baseline timing: recompute scores/weights for the fixed routes.
    topk_weight = (x.float() @ w_router.T).gather(1, topk_idx.long()).softmax(-1)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    try:
        buffer = getattr(args, "_deep_ep_buffer", None)
        if buffer is None:
            buffer = ElasticBuffer(
                group, num_max_tokens_per_rank=args.tokens, hidden=args.hidden,
                num_topk=args.top_k, use_fp8_dispatch=False, explicitly_destroy=True,
            )
            args._deep_ep_buffer = buffer
        num_sms = buffer.get_theoretical_num_sms(args.num_experts, args.top_k)
        recv_x, recv_idx, recv_weights, handle, event = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weight,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=args.tokens,
            expert_alignment=args.expert_alignment,
            num_sms=num_sms,
            async_with_compute_stream=False,
            do_expand=True,
            do_zero_padding=True,
            do_cpu_sync=True,
        )
        if event is not None and hasattr(event, "current_stream_wait"):
            event.current_stream_wait()
        counts = list(handle.num_recv_tokens_per_expert_list)
        local_e = args.num_experts // world
        if len(counts) == args.num_experts:
            counts = counts[rank * local_e : (rank + 1) * local_e]
        if len(counts) != local_e:
            raise RuntimeError(f"DeepEP returned {len(counts)} expert counts, expected {local_e}")
        # In expanded mode DeepEP inserts expert-alignment gaps.  Its psum
        # layout records real per-expert endpoints while preserving those gaps;
        # this is the layout DeepGEMM explicitly accepts with use_psum_layout.
        layout = handle.psum_num_recv_tokens_per_expert.contiguous()
        if layout.numel() != local_e:
            raise RuntimeError("DeepEP returned an invalid per-expert prefix layout")
        if recv_x.shape[0] < int(layout[-1].item() if layout.numel() else 0):
            raise RuntimeError("DeepEP expanded layout is shorter than its metadata")
        expected_m = max(
            1, world * args.tokens * args.top_k // args.num_experts
        )
        # In expand mode rows are expert-major.  DeepGEMM groups only M; N/K
        # are fixed, precisely matching this layout.
        w_gate, w_up, w_down = weights
        gate = torch.empty((recv_x.shape[0], args.ffn), device=device, dtype=torch.bfloat16)
        up = torch.empty_like(gate)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            recv_x, w_gate, gate, layout, use_psum_layout=True,
            expected_m_for_psum_layout=expected_m,
        )
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            recv_x, w_up, up, layout, use_psum_layout=True,
            expected_m_for_psum_layout=expected_m,
        )
        hidden = torch.nn.functional.silu(gate) * up
        out_local = torch.empty((recv_x.shape[0], args.hidden), device=device, dtype=torch.bfloat16)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            hidden, w_down, out_local, layout, use_psum_layout=True,
            expected_m_for_psum_layout=expected_m,
        )
        # combine sums payloads; its optional topk_weights is a separate
        # reduction payload, not an instruction to multiply expert outputs.
        weighted_out = (out_local.float() * recv_weights[:, None]).to(out_local.dtype)
        output, _, event = buffer.combine(weighted_out, handle=handle,
                                         num_sms=num_sms, async_with_compute_stream=False)
        if event is not None and hasattr(event, "current_stream_wait"):
            event.current_stream_wait()
    except Exception as exc:  # pragma: no cover - target-specific extension
        raise _deepseek_error(exc) from exc
    return {"output": output, "x": x, "topk_idx": topk_idx, "topk_weight": topk_weight,
            "weights": weights, "w_router": w_router}


def reference_result(result, args, rank, world, group):
    """Independent local PyTorch autograd, with gathered expert weights.

    Each rank differentiates its local objective. Expert-weight gradients are
    summed across sources and sliced by owner; the router is replicated.
    This is a small-case correctness oracle, not a scalable training method.
    """
    gathered = []
    train = args.phase == "train"
    for tensor in result["weights"]:
        parts = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        gathered.append(torch.cat(parts, dim=0).detach().requires_grad_(train))
    w_gate, w_up, w_down = gathered
    x = result["x"].detach().clone().requires_grad_(train)
    wr = result["w_router"].detach().clone().requires_grad_(train)
    idx = result["topk_idx"].long()
    p = (x.float() @ wr.T).gather(1, idx).softmax(-1)
    if train:
        p.retain_grad()
    # Keep empty experts attached to autograd so their weight gradients are zero.
    output = torch.zeros_like(x, dtype=torch.float32) + sum(w.sum().float() * 0 for w in gathered)
    for e in range(args.num_experts):
        token, slot = torch.where(idx == e)
        a = x[token]
        gate = a @ w_gate[e].T
        up = a @ w_up[e].T
        hidden = torch.nn.functional.silu(gate) * up
        branch = hidden @ w_down[e].T
        output = output.index_add(0, token, branch.float() * p[token, slot, None])
    output = output.to(x.dtype)
    expected = {"output": output.detach()}
    if train:
        (output.float() * result["dout"].float()).sum().backward()
        expected.update(dx=x.grad, dweight=p.grad)
        dwr = wr.grad.float()
        dist.all_reduce(dwr, group=group)
        expected["dw_router"] = dwr
        local_e = args.num_experts // world
        for name, w in zip(("dw_gate", "dw_up", "dw_down"), gathered):
            dw = w.grad.float()
            dist.all_reduce(dw, group=group)
            expected[name] = dw[rank * local_e:(rank + 1) * local_e].contiguous()
    return expected


def comparison_stats(actual, expected, *, rtol=0.03, atol=1e-6):
    """Scale-sensitive checks: an all-zero answer fails for a nonzero reference."""
    if actual.shape != expected.shape:
        return False, float("inf"), float("inf")
    a, b = actual.detach().float(), expected.detach().float()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        return False, float("inf"), float("inf")
    if a.numel() == 0:
        return True, 0.0, 0.0
    delta = a - b
    rel = float(delta.norm() / b.norm().clamp_min(1e-12))
    maximum = float(delta.abs().max())
    ok = rel <= rtol and maximum <= atol + rtol * float(b.abs().max())
    return ok, maximum, rel


def check_result(result, expected, group, rank, label):
    failed = []
    for name, ref in expected.items():
        ok, abs_error, rel_error = comparison_stats(result[name], ref)
        metrics = torch.tensor([int(not ok), abs_error, rel_error], device=ref.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=group)
        fail, abs_error, rel_error = metrics.tolist()
        if rank == 0:
            print(f"[{label}] {name}: max abs={abs_error:.3e}, relative L2={rel_error:.3e}, {'FAIL' if fail else 'ok'}")
        if fail:
            failed.append(name)
    if failed:
        raise RuntimeError(f"{label} failed: {', '.join(failed)}")


def benchmark_runner(runner, args, group):
    # Inputs and library buffers are created before timing. Dynamic dispatch,
    # schedule construction, allocations, compute, and combine remain included.
    for _ in range(args.warmup):
        runner()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.bench_iters):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = runner()
        torch.cuda.synchronize()
        ms = torch.tensor((time.perf_counter() - start) * 1000, device=result["output"].device)
        dist.all_reduce(ms, op=dist.ReduceOp.MAX, group=group)
        samples.append(ms.item())
        del result
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch-triton", "deepseek", "both"), default="torch-triton")
    parser.add_argument("--phase", choices=("train", "prefill", "decode"), default="train")
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--ep-size", type=int, default=None, help="expert-parallel group size; this reference requires it to equal torchrun world size")
    parser.add_argument("--tokens", type=int, default=None, help="tokens per rank; defaults depend on phase")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-alignment", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check", action="store_true", help="check output and, in train, all computed gradients against PyTorch autograd")
    parser.add_argument("--bench-iters", type=int, default=0, help="synchronized end-to-end timing iterations; 0 disables timing")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.phase == "train" and args.backend != "torch-triton":
        parser.error("DeepSeek comparison supports forward only; choose --phase prefill or decode, or torch-triton for train")
    if args.bench_iters < 0 or args.warmup < 1:
        parser.error("bench-iters must be nonnegative and warmup must be positive")
    _require_cuda()
    if args.tokens is None:
        args.tokens = {"train": 256, "prefill": 256, "decode": 8}[args.phase]
    if min(args.tokens, args.hidden, args.ffn, args.num_experts, args.top_k) <= 0:
        parser.error("dimensions and top-k must be positive")
    rank, world, _, group = _init_distributed()
    if args.ep_size is not None and args.ep_size != world:
        raise RuntimeError(
            f"this single-node reference uses one explicit full-world EP group; "
            f"--ep-size must equal torchrun world size ({world})"
        )
    if args.num_experts % world != 0:
        raise RuntimeError(f"num-experts ({args.num_experts}) must be divisible by torchrun world size ({world})")
    if args.top_k > args.num_experts:
        raise RuntimeError("top-k cannot exceed num-experts")
    _dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = _make_inputs(args, rank, world, torch.device("cuda", torch.cuda.current_device()))
    backends = ("torch-triton", "deepseek") if args.backend == "both" else (args.backend,)
    baseline_output = None
    for backend in backends:
        run = run_torch_triton if backend == "torch-triton" else run_deepseek
        runner = lambda: run(args, rank, world, group, inputs=inputs)
        result = runner()
        if args.check:
            expected = reference_result(result, args, rank, world, group)
            check_result(result, expected, group, rank, backend)
            del expected
        if baseline_output is not None:
            check_result(result, {"output": baseline_output}, group, rank, "backend comparison")
        if backend == "torch-triton" and args.backend == "both":
            baseline_output = result["output"].clone()
        if args.bench_iters:
            ms = benchmark_runner(runner, args, group)
            if rank == 0:
                scope = "forward + autograd backward + router gradient SUM" if args.phase == "train" else "router weights + dispatch + expert forward + combine"
                print(f"[{backend}] median max-rank wall time={ms:.3f} ms ({scope}; dynamic preparation included)")
        if rank == 0:
            print(f"backend={backend} phase={args.phase} dtype={args.dtype} world={world} output={tuple(result['output'].shape)}")
        del result
    dist.barrier(group=group)
    if hasattr(args, "_deep_ep_buffer"):
        args._deep_ep_buffer.destroy()
        del args._deep_ep_buffer
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
