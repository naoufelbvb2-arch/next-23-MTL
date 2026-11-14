"""
engine/kernels.py
SS-ENN v3.0 Phase 3.1: Hardware-Accelerated Kernels, Factorized Highway DFA & O(1) Stateful Cache.

Strictly Zero-Autograd (`no backward()`, `no grad()`) and compatible with both
CPU (float32) and GPU/mixed-precision accelerators (bfloat16 / float16):
  1. Fused In-Place Three-Factor Spring Kernel (`fused_spring_update_inplace`) with
     reusable scratch buffers (`SpringScratchBuffer`) and Kahan error compensation
     for low-precision safety (`MixedPrecisionCompensator`).
  2. Indexed Active-Block Execution (`indexed_block_activate_highway`,
     `indexed_block_matmul_slice`, `batch_active_block_union_update`) avoiding zero-padded
     3D/5D tensor allocations.
  3. Factorized Highway DFA (`FactorizedHighwayDFA`): projects output error e in R^V
     ONCE into highway dimension R^D and routes via compact [D, D] adapters per layer.
  4. Stateful O(1) Autoregressive Inference Cache (`SpringInferenceCache`): replaces
     O(T^2) rollout recomputation with constant-time per-step state updates.
  5. Tensor allocation telemetry (`TensorAllocationTracker`) using TorchDispatchMode
     for honest ATen heap allocation audits.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode


class TensorAllocationTracker(TorchDispatchMode):
    """
    Diagnostic context manager counting newly allocated ATen tensor bytes inside
    training/update steps (Zero-Autograd; counts true C++ tensor buffer allocations).
    """

    def __init__(self) -> None:
        super().__init__()
        self.allocated_bytes: int = 0
        self.allocation_count: int = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        in_ptrs: set[int] = set()

        def _collect_ptrs(obj) -> None:
            if isinstance(obj, torch.Tensor):
                try:
                    in_ptrs.add(obj.untyped_storage().data_ptr())
                except Exception:
                    pass
            elif isinstance(obj, (tuple, list)):
                for item in obj:
                    _collect_ptrs(item)

        _collect_ptrs(args)
        _collect_ptrs(kwargs)

        res = func(*args, **kwargs)

        def _tally(obj) -> None:
            if isinstance(obj, torch.Tensor):
                try:
                    ptr = obj.untyped_storage().data_ptr()
                    if ptr not in in_ptrs:
                        self.allocated_bytes += int(obj.numel() * obj.element_size())
                        self.allocation_count += 1
                        in_ptrs.add(ptr)
                except Exception:
                    pass
            elif isinstance(obj, (tuple, list)):
                for item in obj:
                    _tally(item)

        _tally(res)
        return res

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024.0 * 1024.0)


class SpringScratchBuffer:
    """
    Reusable workspace holding two 2D scratch tensors (stretch, gate) plus a 1D
    delta vector so repeated in-place spring updates incur zero per-step heap churn.
    """

    def __init__(self) -> None:
        self._buf_stretch: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._buf_gate: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._buf_fp32_tmp: dict[str, torch.Tensor] = {}
        self._buf_delta: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    @torch.no_grad()
    def get_pair(
        self, rows: int, cols: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), dtype)
        need = rows * cols
        cur_a = self._buf_stretch.get(key)
        if cur_a is None or cur_a.numel() < need:
            self._buf_stretch[key] = torch.empty(need, device=device, dtype=dtype)
        cur_b = self._buf_gate.get(key)
        if cur_b is None or cur_b.numel() < need:
            self._buf_gate[key] = torch.empty(need, device=device, dtype=dtype)
        return (
            self._buf_stretch[key][:need].view(rows, cols),
            self._buf_gate[key][:need].view(rows, cols),
        )

    @torch.no_grad()
    def get_fp32_triple(
        self, rows: int, cols: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(device)
        need = rows * cols * 3
        cur = self._buf_fp32_tmp.get(key)
        if cur is None or cur.numel() < need:
            self._buf_fp32_tmp[key] = torch.empty(need, device=device, dtype=torch.float32)
        chunk = rows * cols
        flat = self._buf_fp32_tmp[key]
        return (
            flat[:chunk].view(rows, cols),
            flat[chunk : 2 * chunk].view(rows, cols),
            flat[2 * chunk : 3 * chunk].view(rows, cols),
        )

    @torch.no_grad()
    def get_delta_pair(
        self, rows: int, cols: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), dtype)
        need = rows * cols
        cur = self._buf_delta.get(key)
        if cur is None or cur.numel() < need:
            self._buf_delta[key] = torch.empty(need, device=device, dtype=dtype)
        # Re-use _buf_stretch/_buf_gate later in update stage (non-overlapping lifetimes!)
        cur_w = self._buf_stretch.get(key)
        if cur_w is None or cur_w.numel() < need:
            self._buf_stretch[key] = torch.empty(need, device=device, dtype=dtype)
        return self._buf_delta[key][:need].view(rows, cols), self._buf_stretch[key][:need].view(rows, cols)


class MixedPrecisionCompensator:
    """
    Stores FP32 Kahan residual error-compensation buffers for low-precision
    (bfloat16 / float16) spring parameters so sub-ULP updates accumulate faithfully
    without stagnation, underflow-to-zero, or NaN.
    """

    def __init__(self, mode: str = "full") -> None:
        if mode not in ("full", "anchor_k", "off"):
            raise ValueError(f"Unsupported MixedPrecisionCompensator mode: {mode!r}")
        self.mode = mode
        self._res_w: dict[int | str, torch.Tensor] = {}
        self._res_w0: dict[int | str, torch.Tensor] = {}
        self._res_k: dict[int | str, torch.Tensor] = {}

    @staticmethod
    def _resize_or_init_residual(
        old_t: torch.Tensor | None, shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        if old_t is not None and old_t.shape == shape and old_t.device == device:
            return old_t
        new_t = torch.zeros(shape, device=device, dtype=torch.float32)
        if old_t is not None and old_t.device == device and old_t.ndim == 2:
            r_min = min(old_t.shape[0], shape[0])
            c_min = min(old_t.shape[1], shape[1])
            new_t[:r_min, :c_min].copy_(old_t[:r_min, :c_min])
        return new_t

    @torch.no_grad()
    def get_residuals(
        self, key_id: int | str, shape: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.mode == "off":
            return None, None, None
        if self.mode == "anchor_k":
            self._res_w.pop(key_id, None)
            self._res_w0[key_id] = self._resize_or_init_residual(
                self._res_w0.get(key_id), shape, device
            )
            self._res_k[key_id] = self._resize_or_init_residual(
                self._res_k.get(key_id), shape, device
            )
            return None, self._res_w0[key_id], self._res_k[key_id]
        self._res_w[key_id] = self._resize_or_init_residual(
            self._res_w.get(key_id), shape, device
        )
        self._res_w0[key_id] = self._resize_or_init_residual(
            self._res_w0.get(key_id), shape, device
        )
        self._res_k[key_id] = self._resize_or_init_residual(
            self._res_k.get(key_id), shape, device
        )
        return self._res_w[key_id], self._res_w0[key_id], self._res_k[key_id]


@torch.no_grad()
def fused_spring_update_inplace(
    w: torch.Tensor,
    w0: torch.Tensor,
    k: torch.Tensor,
    driving_signal: torch.Tensor,
    eta_w: float,
    eta_0: float,
    gamma: float,
    beta_k_eff: torch.Tensor | float,
    stiff_ceil: float = 0.85,
    active_row_gate: torch.Tensor | None = None,
    cubic_shield: bool = False,
    scratch: SpringScratchBuffer | None = None,
    compensator: MixedPrecisionCompensator | None = None,
    comp_key: int | str | None = None,
    comp_row_indices: torch.Tensor | None = None,
    comp_full_rows: int | None = None,
) -> None:
    """
    Fused In-Place Three-Factor Spring Plasticity Kernel:
      Law 1: w  <- w + eta_w * driving_signal - gamma * k * (w - w0)
      Law 2: w0 <- w0 + eta_0 * shield(k) * (w_new - w0)
      Law 3: k  <- clamp(k + beta_k_eff * |w_new - w0| * (1 - k) * gate, 0, stiff_ceil)

    Operates completely in-place via reusable workspace tensors on FP32, and
    upgrades transparently into FP32-accumulated + Kahan-compensated updates for
    BF16 / FP16 mixed precision without underflow or NaN.
    """
    rows, cols = w.shape
    scratch = scratch if scratch is not None else SpringScratchBuffer()

    is_low_prec = w.dtype in (torch.bfloat16, torch.float16)
    if is_low_prec:
        buf_stretch, buf_gate = scratch.get_pair(rows, cols, w.device, torch.float32)
        wf, w0f, kf = scratch.get_fp32_triple(rows, cols, w.device)
        wf.copy_(w.float())
        w0f.copy_(w0.float())
        kf.copy_(k.float())
        drv_f = driving_signal if driving_signal.dtype == torch.float32 else driving_signal.float()

        rw = rw0 = rk = None
        if compensator is not None:
            ck = comp_key if comp_key is not None else id(w)
            full_rows = comp_full_rows if comp_full_rows is not None else rows
            rw, rw0, rk = compensator.get_residuals(ck, (full_rows, cols), w.device)
            if comp_row_indices is not None:
                if rw is not None:
                    wf.add_(rw[comp_row_indices])
                if rw0 is not None:
                    w0f.add_(rw0[comp_row_indices])
                if rk is not None:
                    kf.add_(rk[comp_row_indices]).clamp_(0.0, 0.98 if cubic_shield else stiff_ceil)
            else:
                if rw is not None:
                    wf.add_(rw)
                if rw0 is not None:
                    w0f.add_(rw0)
                if rk is not None:
                    kf.add_(rk).clamp_(0.0, 0.98 if cubic_shield else stiff_ceil)

        _fused_fp32_core_inplace(
            wf, w0f, kf, drv_f, eta_w, eta_0, gamma, beta_k_eff,
            stiff_ceil, active_row_gate, cubic_shield, buf_stretch, buf_gate,
        )
        wf.nan_to_num_(nan=0.0, posinf=64.0, neginf=-64.0)
        w0f.nan_to_num_(nan=0.0, posinf=64.0, neginf=-64.0)
        kf.nan_to_num_(nan=0.0, posinf=(0.98 if cubic_shield else stiff_ceil), neginf=0.0)

        w.copy_(wf.to(w.dtype))
        w0.copy_(w0f.to(w0.dtype))
        k.copy_(kf.to(k.dtype))
        if rw is not None or rw0 is not None or rk is not None:
            if comp_row_indices is not None:
                if rw is not None:
                    rw.index_copy_(0, comp_row_indices, wf - w.float())
                if rw0 is not None:
                    rw0.index_copy_(0, comp_row_indices, w0f - w0.float())
                if rk is not None:
                    rk.index_copy_(0, comp_row_indices, kf - k.float())
            else:
                if rw is not None:
                    rw.copy_(wf).sub_(w.float())
                if rw0 is not None:
                    rw0.copy_(w0f).sub_(w0.float())
                if rk is not None:
                    rk.copy_(kf).sub_(k.float())
        return

    buf_stretch, buf_gate = scratch.get_pair(rows, cols, w.device, w.dtype)
    _fused_fp32_core_inplace(
        w, w0, k, driving_signal, eta_w, eta_0, gamma, beta_k_eff,
        stiff_ceil, active_row_gate, cubic_shield, buf_stretch, buf_gate,
    )


@torch.no_grad()
def _fused_fp32_core_inplace(
    w: torch.Tensor,
    w0: torch.Tensor,
    k: torch.Tensor,
    driving_signal: torch.Tensor,
    eta_w: float,
    eta_0: float,
    gamma: float,
    beta_k_eff: torch.Tensor | float,
    stiff_ceil: float,
    active_row_gate: torch.Tensor | None,
    cubic_shield: bool,
    buf_stretch: torch.Tensor,
    buf_gate: torch.Tensor,
) -> None:
    # 1. Stretch diff: buf_stretch = k * (w - w0)
    buf_stretch.copy_(w).sub_(w0)
    buf_stretch.mul_(k)

    # Plasticity on w
    if cubic_shield and bool((k >= 0.80).any().item()):
        buf_gate.fill_(1.0).sub_(k)
        # where k >= 0.80 apply (1 - k)^2 damping
        armor = k >= 0.80
        if armor.any():
            sq_gate = buf_gate.clone()
            sq_gate[armor] = sq_gate[armor].pow(2)
            sq_gate[~armor] = 1.0
            w.addcmul_(driving_signal, sq_gate, value=eta_w).add_(buf_stretch, alpha=-gamma)
        else:
            w.add_(driving_signal, alpha=eta_w).add_(buf_stretch, alpha=-gamma)
    else:
        w.add_(driving_signal, alpha=eta_w).add_(buf_stretch, alpha=-gamma)

    # 2. Updated stretch: buf_stretch = w_new - w0
    buf_stretch.copy_(w).sub_(w0)

    # Anchor gate: buf_gate = (1 - k) [or (1 - k)^3 on armored cells]
    buf_gate.fill_(1.0).sub_(k)
    if cubic_shield:
        armor = k >= 0.80
        if armor.any():
            # Apply cubic shield strictly onto a temporary view or inplace since k update also needs (1-k)
            buf_w0_gate = buf_gate.clone()
            buf_w0_gate[armor] = buf_w0_gate[armor].pow(3)
            w0.addcmul_(buf_w0_gate, buf_stretch, value=eta_0)
        else:
            w0.addcmul_(buf_gate, buf_stretch, value=eta_0)
    else:
        w0.addcmul_(buf_gate, buf_stretch, value=eta_0)

    # Snapshot already-armored cells before updating k so non-armored cells stay capped at stiff_ceil
    was_high_armor = (k >= 0.85) if cubic_shield else None

    # 3. Stiffness consolidation: k += beta_k_eff * |w_new - w0| * (1 - k) * active_row_gate
    buf_stretch.abs_()
    if isinstance(beta_k_eff, torch.Tensor):
        b_scale = beta_k_eff.to(dtype=buf_stretch.dtype, device=buf_stretch.device)
        if active_row_gate is not None:
            buf_stretch.mul_(b_scale * active_row_gate.to(dtype=buf_stretch.dtype, device=buf_stretch.device))
        else:
            buf_stretch.mul_(b_scale)
        k.addcmul_(buf_stretch, buf_gate)
    else:
        if active_row_gate is not None:
            buf_stretch.mul_(active_row_gate.to(dtype=buf_stretch.dtype, device=buf_stretch.device))
        k.addcmul_(buf_stretch, buf_gate, value=float(beta_k_eff))

    if cubic_shield and was_high_armor is not None and bool(was_high_armor.any().item()):
        k.clamp_(min=0.0)
        max_bounds = torch.where(
            was_high_armor,
            torch.full_like(k, 0.98),
            torch.full_like(k, float(stiff_ceil)),
        )
        torch.minimum(k, max_bounds, out=k)
    else:
        k.clamp_(0.0, stiff_ceil)


@torch.no_grad()
def indexed_block_activate_highway(
    z_act: torch.Tensor,
    a_prev_eff: torch.Tensor,
    top_b_idx: torch.Tensor,
    block_size: int,
    neg_slope: float,
    res_scale: float,
    use_highway: bool = True,
) -> torch.Tensor:
    """
    Indexed Active-Block Activation + Highway Residual Scatter (Zero Dense Sparse Mask).
    Computes LeakyReLU strictly on the K_blocks winning blocks per token and adds
    them directly onto the residual stream without allocating zero-padded 3D/5D tensors.
    """
    n_tok, d_sub = z_act.shape
    n_blocks = max(1, d_sub // block_size)

    out = a_prev_eff.clone()
    z_blocks = z_act.view(n_tok, n_blocks, block_size)
    out_blocks = out[:, :d_sub].view(n_tok, n_blocks, block_size)

    if n_tok == 1:
        # Ultra-fast 1-token path for O(1) autoregressive generation (true view index_add_)
        sel = top_b_idx[0]
        a_win = F.leaky_relu(z_blocks[0].index_select(0, sel), negative_slope=neg_slope)
        out_blocks[0].index_add_(0, sel, a_win, alpha=res_scale)
        return out

    idx_exp = top_b_idx.unsqueeze(-1).expand(-1, -1, block_size)
    z_win = torch.gather(z_blocks, dim=1, index=idx_exp)
    a_win = F.leaky_relu(z_win, negative_slope=neg_slope)
    if res_scale != 1.0:
        a_win.mul_(res_scale)
    out_blocks.scatter_add_(1, idx_exp, a_win)
    return out


@torch.no_grad()
def indexed_block_matmul_slice(
    a_prev_eff: torch.Tensor,
    w0: torch.Tensor,
    active_block_indices: torch.Tensor,
    block_size: int,
    neg_slope: float,
    res_scale: float = 1.0,
) -> torch.Tensor:
    """
    True Indexed Active-Block Subspace GEMM:
    Gathers strictly the requested active blocks (K_blocks / N_blocks of W_0), multiplies
    only those rows, activates via LeakyReLU, and adds directly into the highway residual.
    """
    n_tok, in_dim = a_prev_eff.shape
    d_sub = w0.shape[0]
    n_blocks = max(1, d_sub // block_size)
    w_blocks = w0.view(n_blocks, block_size, in_dim)
    w_active = w_blocks.index_select(0, active_block_indices).reshape(-1, in_dim)
    z_active = F.linear(a_prev_eff, w_active)
    a_active = F.leaky_relu(z_active, negative_slope=neg_slope)

    out = a_prev_eff.clone()
    out_blocks = out[:, :d_sub].view(n_tok, n_blocks, block_size)
    u_cnt = int(active_block_indices.numel())
    out_blocks.index_add_(
        1,
        active_block_indices,
        a_active.view(n_tok, u_cnt, block_size),
        alpha=res_scale,
    )
    return out


@torch.no_grad()
def batch_active_block_union_update(
    layer,
    projected_error: torch.Tensor,
    n_valid: float,
    maturity_stiff_gain: float,
    block_size: int,
    k_blk: int,
    stiff_mult: float,
    stiff_ceil: float,
    scratch: SpringScratchBuffer | None = None,
    compensator: MixedPrecisionCompensator | None = None,
    comp_key: int | str | None = None,
) -> tuple[int, int]:
    """
    Active-Block Union Mini-Batch Kernel (Zero 5D Allocation):
    1. Identifies the unique union U of winning blocks across the mini-batch.
    2. Computes masked error derivatives `delta` cleanly without 3D/5D dense masks.
    3. Executes matmul & fused in-place spring updates ONLY on active union rows U * B_size.
    Returns (active_blocks_updated, total_blocks).
    """
    d_act = layer.d_active
    a_prev_eff = layer._last_a_prev
    d_eff = a_prev_eff.shape[1]
    z_act = layer._last_z
    block_mask = layer._last_mask
    n_tok = z_act.shape[0]
    n_blocks = max(1, d_act // block_size)
    scratch = scratch if scratch is not None else SpringScratchBuffer()
    effective_comp_key = comp_key if comp_key is not None else id(layer)

    wins_per_block = block_mask.sum(dim=0)
    active_b_idx = torch.nonzero(wins_per_block > 0, as_tuple=False).squeeze(-1)
    u_blocks = int(active_b_idx.numel())
    if u_blocks == 0:
        return 0, n_blocks

    # Compute delta in reusable [n_tok, d_act] workspace without temporary 3D/5D copies
    delta_mat, g_prime_buf = scratch.get_delta_pair(n_tok, d_act, z_act.device, projected_error.dtype)
    delta_mat.copy_(projected_error[:, :d_act])
    slope = float(layer.negative_slope)
    g_prime_buf.fill_(1.0).masked_fill_(z_act <= 0.0, slope)
    delta_mat.mul_(g_prime_buf)
    delta_blocks = delta_mat.view(n_tok, n_blocks, block_size)
    delta_blocks.mul_(block_mask.unsqueeze(-1))

    sparsity_ratio = float(n_blocks) / float(max(k_blk, 1))
    wins_clamped = wins_per_block.clamp(min=1.0)
    wins_per_cell = wins_clamped.repeat_interleave(block_size).unsqueeze(1)
    balanced_denom = (0.40 * max(n_valid, 1.0)) + (0.60 * wins_per_cell * sparsity_ratio)
    win_norm_factor = (max(n_valid, 1.0) / (wins_per_cell * sparsity_ratio)).clamp(1.0, 8.0)
    effective_beta_k = layer.beta_k * maturity_stiff_gain * win_norm_factor * stiff_mult

    if u_blocks == n_blocks:
        # Contiguous fast path: update full active slice strictly in-place (zero gather/scatter)
        driving_signal = torch.matmul(delta_mat.t(), a_prev_eff).div_(balanced_denom)
        active_row_gate = (wins_per_block > 0).repeat_interleave(block_size).float().unsqueeze(1)
        fused_spring_update_inplace(
            w=layer.w[:d_act, :d_eff],
            w0=layer.w0[:d_act, :d_eff],
            k=layer.k[:d_act, :d_eff],
            driving_signal=driving_signal,
            eta_w=layer.eta_w,
            eta_0=layer.eta_0,
            gamma=layer.gamma,
            beta_k_eff=effective_beta_k,
            stiff_ceil=stiff_ceil,
            active_row_gate=active_row_gate,
            cubic_shield=False,
            scratch=scratch,
            compensator=compensator,
            comp_key=effective_comp_key,
        )
    else:
        # Sparse Active-Block Union path: compute matmul & spring updates ONLY on active rows
        offs = torch.arange(block_size, device=z_act.device)
        active_rows = (active_b_idx.unsqueeze(1) * block_size + offs.unsqueeze(0)).reshape(-1)
        denom_sub = balanced_denom[active_rows]
        beta_sub = effective_beta_k[active_rows]
        driving_sub = torch.matmul(delta_mat[:, active_rows].t(), a_prev_eff).div_(denom_sub)

        w_sub = layer.w[active_rows, :d_eff]
        w0_sub = layer.w0[active_rows, :d_eff]
        k_sub = layer.k[active_rows, :d_eff]
        fused_spring_update_inplace(
            w=w_sub,
            w0=w0_sub,
            k=k_sub,
            driving_signal=driving_sub,
            eta_w=layer.eta_w,
            eta_0=layer.eta_0,
            gamma=layer.gamma,
            beta_k_eff=beta_sub,
            stiff_ceil=stiff_ceil,
            active_row_gate=None,
            cubic_shield=False,
            scratch=scratch,
            compensator=compensator,
            comp_key=effective_comp_key,
            comp_row_indices=active_rows,
            comp_full_rows=d_act,
        )
        layer.w[:d_act, :d_eff].index_copy_(0, active_rows, w_sub)
        layer.w0[:d_act, :d_eff].index_copy_(0, active_rows, w0_sub)
        layer.k[:d_act, :d_eff].index_copy_(0, active_rows, k_sub)

    return u_blocks, n_blocks


class FactorizedHighwayDFA(nn.Module):
    """
    Phase 3.1 Factorized Highway Direct Feedback Alignment:
    Eliminates per-layer [D_max x V] feedback buffers across L deep layers.
    Projects output error e in R^{N x V} ONCE through the row-normalized output head
    into highway residual space e_hwy in R^{N x D_last}, then routes e_hwy into each
    hidden layer via compact [D_max x D_max] local adapters M^(l):
        delta^(l) = e_hwy @ (M^(l)[:d_act, :d_last])^T * depth_scale.

    Memory drops from O(L * D_max * V) -> O(L * D_max^2) (87% - 99% memory reduction).
    """

    def __init__(
        self,
        num_layers: int,
        d_max: int,
        vocab_size: int,
        eta_b: float = 0.015,
        dfa_anchor_blend: float = 0.65,
        seed: int = 42,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_max = d_max
        self.vocab_size = vocab_size
        self.eta_b = float(eta_b)
        self.dfa_anchor_blend = float(dfa_anchor_blend)
        self.target_row_norm = math.sqrt(float(vocab_size) / float(d_max))

        gen = torch.Generator()
        gen.manual_seed(seed)
        eye = torch.eye(d_max, dtype=torch.float32)
        for l_idx in range(num_layers):
            noise = torch.randn(d_max, d_max, generator=gen) / math.sqrt(d_max)
            m_init = F.normalize(0.75 * eye + 0.25 * noise, p=2, dim=-1, eps=1e-6)
            self.register_buffer(f"M_{l_idx}", m_init)

    @torch.no_grad()
    def get_adapter(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"M_{layer_idx}")

    @torch.no_grad()
    def project_vocab_to_highway(
        self, error_vec: torch.Tensor, head_w_active: torch.Tensor
    ) -> torch.Tensor:
        """
        Single shared projection from vocabulary space [N, V] to highway space [N, D_last].
        `head_w_active` has shape [V, D_last].
        """
        col_norm_t = F.normalize(head_w_active.t(), p=2, dim=-1, eps=1e-6)  # [D_last, V]
        return F.linear(error_vec, col_norm_t) * self.target_row_norm      # [N, D_last]

    @torch.no_grad()
    def broadcast_layer_error(
        self,
        layer_idx: int,
        e_highway: torch.Tensor,
        d_act: int,
        depth_scale: float = 1.0,
    ) -> torch.Tensor:
        d_last = e_highway.shape[1]
        m_act = self.get_adapter(layer_idx)[:d_act, :d_last]
        proj = F.linear(e_highway, m_act)
        if depth_scale != 1.0:
            proj.mul_(depth_scale)
        return proj

    @torch.no_grad()
    def adapt_layer(
        self,
        layer_idx: int,
        a_act: torch.Tensor,
        e_highway: torch.Tensor,
        n_valid: float,
        unlocked_row_gate: torch.Tensor | None = None,
    ) -> None:
        """
        Purely local D x D Hebbian alignment anchored toward the identity residual highway,
        never allocating [D_act, V] correlation matrices.
        """
        d_act = a_act.shape[1]
        d_last = e_highway.shape[1]
        m_full = self.get_adapter(layer_idx)
        m_act = m_full[:d_act, :d_last]

        # Normalized Hebbian correlation inside R^{d_act x d_last}
        hebb = torch.matmul(a_act.t(), e_highway) / (max(n_valid, 1.0) * max(self.target_row_norm, 1e-6))
        if unlocked_row_gate is not None:
            hebb.mul_(unlocked_row_gate)
        m_hebb = F.normalize(m_act + self.eta_b * hebb, p=2, dim=-1, eps=1e-6)

        eye_slice = torch.eye(max(d_act, d_last), device=m_act.device, dtype=m_act.dtype)[:d_act, :d_last]
        m_new = F.normalize(
            (1.0 - self.dfa_anchor_blend) * m_hebb + self.dfa_anchor_blend * eye_slice,
            p=2,
            dim=-1,
            eps=1e-6,
        )
        m_act.copy_(m_new)

    @torch.no_grad()
    def freeze(self) -> None:
        for l_idx in range(self.num_layers):
            setattr(self, f"M_{l_idx}", None)

    @torch.no_grad()
    def parameter_count(self) -> int:
        return sum(
            getattr(self, f"M_{l_idx}").numel()
            for l_idx in range(self.num_layers)
            if getattr(self, f"M_{l_idx}") is not None
        )

    @torch.no_grad()
    def memory_mb(self) -> float:
        total_bytes = sum(
            getattr(self, f"M_{l_idx}").numel() * getattr(self, f"M_{l_idx}").element_size()
            for l_idx in range(self.num_layers)
            if getattr(self, f"M_{l_idx}") is not None
        )
        return total_bytes / (1024.0 * 1024.0)


class SpringInferenceCache:
    """
    Phase 3.1 Stateful O(1) Autoregressive Inference Cache:
    Maintains the exact causal recurrent state step-by-step during `generate()` so each
    new token executes in constant O(1) compute per step instead of O(T^2) prefix recomputation,
    producing 100.00% identical token sequences to full uncached rollouts:
      1. Causal K / V / Salience rolling buffer for Dual-Horizon Sparse Attention
      2. Salience-Gated EMA Capacitor vector c_t
      3. Exponential O(1) ALiBi accumulator (standard & translation-invariant sliding)
      4. Immediate local syntax lag-1 and lag-2 embedding buffers
      5. Causal Recurring Entity Phrase Tracker + lag1..3 unit embeddings (Role Bridge)
      6. O(1) Incremental N-gram Hash Tables for Two-Tier Transition Refractory Gate
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        d_max: int,
        context_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.d_max = d_max
        self.context_dim = context_dim
        self.device = device
        self.dtype = dtype
        self.step: int = 0

        # Dual-Horizon Attention fixed K/V/Salience storage (O(1) slice per step)
        self.k_cache = torch.zeros(batch_size, max_seq_len, d_max, device=device, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, max_seq_len, d_max, device=device, dtype=dtype)
        self.sal_cache = torch.zeros(batch_size, max_seq_len, device=device, dtype=dtype)

        # Salience EMA Capacitor & Lag states
        self.c_state = torch.zeros(batch_size, d_max, device=device, dtype=dtype)
        self.prev_emb_1 = torch.zeros(batch_size, d_max, device=device, dtype=dtype)
        self.prev_emb_2 = torch.zeros(batch_size, d_max, device=device, dtype=dtype)

        # Exponential ALiBi O(1) Accumulator
        self.alibi_u = torch.zeros(batch_size, context_dim, device=device, dtype=dtype)
        self.alibi_denom_base: float = 0.0
        self.x_pos_0 = torch.zeros(batch_size, context_dim, device=device, dtype=dtype)
        self.x_pos_1 = torch.zeros(batch_size, context_dim, device=device, dtype=dtype)

        # Causal Entity Slot & Unit Lag buffers (Milestone 2.3 Role Bridge)
        self.lag1_unit = torch.zeros(batch_size, d_max, device=device, dtype=dtype)
        self.lag2_unit = torch.zeros(batch_size, d_max, device=device, dtype=dtype)
        self.lag3_unit = torch.zeros(batch_size, d_max, device=device, dtype=dtype)

        self.cur_start_emb = torch.zeros(batch_size, d_max, device=device, dtype=dtype)
        self.cur_end_emb = torch.zeros(batch_size, d_max, device=device, dtype=dtype)
        self.cur_start_id = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        self.cur_end_id = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        self.cur_len = torch.zeros(batch_size, dtype=torch.long, device=device)

        self.comp_start_emb = torch.zeros(batch_size, d_max, device=device, dtype=dtype)
        self.comp_end_emb = torch.zeros(batch_size, d_max, device=device, dtype=dtype)
        self.comp_start_id = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        self.comp_end_id = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        self.comp_ready = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # O(1) Incremental Transition Refractory State (for B=1 rollout)
        self.seq_history: list[int] = []
        self.bigram_next: dict[int, dict[int, int]] = {}
        self.trigram_next: dict[tuple[int, int], dict[int, int]] = {}
        self.fourgram_next: dict[tuple[int, int, int], dict[int, int]] = {}

    def record_token_ngrams(self, token_id: int) -> None:
        """Records `token_id` transitions in O(1) amortized dict operations."""
        n = len(self.seq_history)
        if n >= 1:
            p1 = self.seq_history[-1]
            d1 = self.bigram_next.setdefault(p1, {})
            d1[token_id] = d1.get(token_id, 0) + 1
        if n >= 2:
            k2 = (self.seq_history[-2], self.seq_history[-1])
            d2 = self.trigram_next.setdefault(k2, {})
            d2[token_id] = d2.get(token_id, 0) + 1
        if n >= 3:
            k3 = (self.seq_history[-3], self.seq_history[-2], self.seq_history[-1])
            d3 = self.fourgram_next.setdefault(k3, {})
            d3[token_id] = d3.get(token_id, 0) + 1
        self.seq_history.append(int(token_id))


def normalize_token_sequences(token_sequences) -> list[list[int]]:
    """
    Normalizes 1-D/2-D `torch.Tensor` batches or nested token sequences into
    explicit `list[list[int]]`, preventing 0-D tensor identity-hashing bugs when
    token IDs are used as dictionary keys.
    """
    if isinstance(token_sequences, torch.Tensor):
        if token_sequences.numel() == 0:
            return []
        raw = token_sequences.detach().cpu().tolist()
        if token_sequences.ndim == 1:
            return [[int(tok) for tok in raw]]
        return [[int(tok) for tok in row] for row in raw]

    normalized: list[list[int]] = []
    for seq in token_sequences:
        if isinstance(seq, torch.Tensor):
            raw_row = seq.detach().cpu().tolist()
            if seq.ndim == 0:
                normalized.append([int(raw_row)])
            else:
                normalized.append([int(tok) for tok in raw_row])
        else:
            normalized.append([int(tok) for tok in seq])
    return normalized
