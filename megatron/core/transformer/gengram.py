# Copyright (c) 2026.
# SPDX-License-Identifier: Apache-2.0

"""Gengram: a lightweight n-gram retrieval residual for Megatron-LM.

This is a simplified, *DNA-token* adaptation of the Engram demo code:
- No hashing / multi-head buckets.
- Direct n-gram table lookup for tokens in {A,T,C,G,N}.
- If any token in the n-gram window is not one of the allowed DNA tokens,
  the lookup is skipped (the retrieved vector is zeroed for that position).

The module is designed to plug into standard Megatron transformer layers where
hidden_states are shaped [s, b, h] and input_ids are shaped [b, s].

Notes:
- The n-gram tables are tiny (5^n), so we keep them replicated across TP ranks.
- For sequence-parallel models, we slice input_ids to the local sequence chunk.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from megatron.core import parallel_state



class DepthwiseShortConv1D(nn.Module):
    """A depthwise 1D conv over the sequence dimension for [s, b, h] tensors."""

    def __init__(self, hidden_size: int, kernel_size: int = 4, dilation: int = 1, activation: bool = True):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.activation = bool(activation)

        # For causal-ish behavior, we pad on the left with (k-1)*d and then crop.
        padding = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=padding,
            groups=self.hidden_size,
            bias=False,
        )
        self.act = nn.SiLU() if self.activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [s, b, h] -> [b, h, s]
        s, b, h = x.shape
        y = self.conv(x.permute(1, 2, 0).contiguous())
        # Crop to original length.
        y = y[:, :, :s]
        if self.act is not None:
            y = self.act(y)
        return y.permute(2, 0, 1).contiguous()  # [s, b, h]


class Gengram(nn.Module):
    """Direct DNA n-gram retrieval residual block."""

    def __init__(
        self,
        *,
        hidden_size: int,
        dna_token_ids: Sequence[int],
        ngram_sizes: Sequence[int],
        embed_dim_per_ngram: int,
        active: bool,
        use_short_conv: bool,
        short_conv_kernel_size: int,
        short_conv_dilation: int,
        window_size: int,
    ):
        super().__init__()

        # Whether this layer should *apply* Engram in forward.
        # Note: parameters still exist (for checkpoint consistency across layers).
        self.active = bool(active)

        self.hidden_size = int(hidden_size)
        self.dna_token_ids = [int(x) for x in dna_token_ids]
        self.ngram_sizes = [int(n) for n in ngram_sizes]
        self.embed_dim_per_ngram = int(embed_dim_per_ngram)

        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError(f"gengram_window_size must be >=1, got {self.window_size}")

        if len(self.dna_token_ids) != 5:
            raise ValueError(
                f"Gengram expects exactly 5 dna_token_ids (A,C,G,T,N order arbitrary), got {self.dna_token_ids}."
            )
        if any(n < 1 for n in self.ngram_sizes):
            raise ValueError(f"ngram_sizes must be >=1, got {self.ngram_sizes}")

        # One embedding table per n.
        self.embeddings = nn.ModuleList()
        for n in self.ngram_sizes:
            num_embeddings = 5 ** n
            self.embeddings.append(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.embed_dim_per_ngram))

        engram_feat_dim = self.embed_dim_per_ngram * len(self.ngram_sizes)
        self.key_proj = nn.Linear(engram_feat_dim, self.hidden_size, bias=False)
        self.value_proj = nn.Linear(engram_feat_dim, self.hidden_size, bias=False)

        self.norm_q = nn.RMSNorm(self.hidden_size, eps=1e-5)
        self.norm_k = nn.RMSNorm(self.hidden_size, eps=1e-5)

        self.use_short_conv = bool(use_short_conv)
        if self.use_short_conv:
            self.short_conv = DepthwiseShortConv1D(
                hidden_size=self.hidden_size,
                kernel_size=short_conv_kernel_size,
                dilation=short_conv_dilation,
                activation=True,
            )
        else:
            self.short_conv = None

        # Precompute base powers [1,5,25,...] up to max n
        max_n = max(self.ngram_sizes) if len(self.ngram_sizes) > 0 else 1
        if self.window_size < max_n:
            raise ValueError(
                f"gengram_window_size ({self.window_size}) must be >= max(ngram_sizes) ({max_n})"
            )
        base_pows = [1]
        for _ in range(1, max_n):
            base_pows.append(base_pows[-1] * 5)
        self.register_buffer("base_pows", torch.tensor(base_pows, dtype=torch.int64), persistent=False)

    def _slice_for_sequence_parallel(self, input_ids: torch.Tensor, local_seq_len: int) -> torch.Tensor:
        """Slice [b, s] input_ids to match local [s_local, b, h] hidden_states when sequence_parallel is on."""
        # If not sequence-parallel or TP size 1, do nothing.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        if world_size == 1:
            return input_ids

        # Megatron sequence parallel typically chunks sequence evenly across TP ranks.
        rank = parallel_state.get_tensor_model_parallel_rank()
        start = rank * local_seq_len
        end = start + local_seq_len
        if input_ids.size(1) < end:
            raise ValueError(
                f"input_ids sequence length ({input_ids.size(1)}) is smaller than required slice end ({end})."
            )
        return input_ids[:, start:end]

    def _compute_ngram_ids(self, small_ids: torch.Tensor, n: int) -> torch.Tensor:
        """Compute n-gram ids for each position using current and previous tokens.

        small_ids: [s, b] with values 0..4 for valid DNA tokens, else -1.
        Returns:
          ngram_ids: [s, b] int64 in [0, 5**n-1], with invalid positions set to 0.
          valid_mask: [s, b] bool True if all tokens in window are valid.
        """
        assert n >= 1
        s, b = small_ids.shape
        device = small_ids.device

        # Collect shifted token ids: t0 (current), t1 (prev1), ...
        toks = []
        toks.append(small_ids)
        for k in range(1, n):
            shifted = torch.full_like(small_ids, -1)
            shifted[k:, :] = small_ids[: s - k, :]
            toks.append(shifted)

        valid = torch.ones((s, b), dtype=torch.bool, device=device)
        for t in toks:
            valid &= t >= 0

        # Build id = sum_k toks[k] * 5^k
        base = self.base_pows[:n].to(device=device)
        out = torch.zeros((s, b), dtype=torch.int64, device=device)
        for k in range(n):
            out += toks[k].clamp_min(0).to(torch.int64) * base[k]

        out = torch.where(valid, out, torch.zeros_like(out))
        return out, valid


    def _pool_windowed_ngram_embeddings(
        self, e_end: torch.Tensor, valid_end: torch.Tensor, n: int
    ) -> torch.Tensor:
        # Pool (average) all n-gram embeddings fully contained in the last W tokens.
        # For a token window of length W ending at position t, valid n-grams of size n
        # correspond to endings in the last L = W - n + 1 positions.
        L = int(self.window_size) - int(n) + 1
        if L <= 0:
            raise ValueError(
                f"gengram_window_size ({self.window_size}) must be >= n ({n}) so that L=W-n+1 >= 1"
            )
        if L == 1:
            return e_end

        # Compute sliding-window sums via cumsum: sum[t] = prefix[t] - prefix[t-L]
        prefix = torch.cumsum(e_end, dim=0)
        prefix_shift = torch.zeros_like(prefix)
        prefix_shift[L:, :, :] = prefix[:-L, :, :]
        sum_win = prefix - prefix_shift

        # Window count of valid endings (float32 for stability), then divide.
        valid_f = valid_end.to(dtype=torch.float32)
        prefix_c = torch.cumsum(valid_f, dim=0)
        prefix_c_shift = torch.zeros_like(prefix_c)
        prefix_c_shift[L:, :] = prefix_c[:-L, :]
        cnt = prefix_c - prefix_c_shift  # [s, b]

        denom = cnt.clamp_min(1.0).to(dtype=e_end.dtype).unsqueeze(-1)
        return sum_win / denom

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute Gengram residual.

        Args:
          hidden_states: [s, b, h]
          input_ids: [b, s_global] (or [b, s_local])
        Returns:
          residual: [s, b, h]
        """
        if not self.active:
            # Return a zero residual without touching input_ids.
            return hidden_states.new_zeros(hidden_states.shape)

        if input_ids is None:
            raise ValueError("Gengram requires input_ids, got None")

        s, b, h = hidden_states.shape
        if h != self.hidden_size:
            raise ValueError(f"hidden_size mismatch: hidden_states has {h}, module has {self.hidden_size}")

        # Ensure input_ids is on same device (it is typically already).
        if input_ids.device != hidden_states.device:
            input_ids = input_ids.to(device=hidden_states.device, non_blocking=True)

        # Slice input ids if sequence-parallel.
        input_ids_local = self._slice_for_sequence_parallel(input_ids, local_seq_len=s)

        if input_ids_local.shape != (b, s):
            # Allow [b, s] only.
            raise ValueError(
                f"input_ids_local shape mismatch: expected ({b},{s}) but got {tuple(input_ids_local.shape)}"
            )

        # Convert to [s, b] token ids.
        toks = input_ids_local.transpose(0, 1).contiguous()  # [s, b]

        # Map to small ids in 0..4 (DNA), else -1.
        small_ids = torch.full_like(toks, -1, dtype=torch.int64)
        for sid, tok_id in enumerate(self.dna_token_ids):
            small_ids = torch.where(toks == tok_id, torch.full_like(small_ids, sid), small_ids)
        # Build concatenated n-gram embeddings with *windowed* pooling.
        # For each position t, we pool all n-grams fully contained in the last W tokens
        # window [t-W+1, t].
        feats = []
        W = int(self.window_size)
        for emb, n in zip(self.embeddings, self.ngram_sizes):
            ngram_ids, valid = self._compute_ngram_ids(small_ids, n=n)
            e_end = emb(ngram_ids)  # [s, b, d]
            e_end = e_end * valid.to(dtype=e_end.dtype).unsqueeze(-1)

            # Number of n-gram *ending positions* that fit in a W-token window.
            # If window is W tokens, n-grams fully inside it correspond to the last (W-n+1) endings.
            L_end = W - int(n) + 1
            if L_end < 1:
                # Should be prevented by config validation, but keep a safe fallback.
                raise ValueError(f"gengram_window_size ({W}) must be >= n ({n})")

            if L_end == 1:
                e_win = e_end
            else:
                # Sliding-window sum via prefix sums: sum_{i=t-L_end+1..t} e_end[i]
                prefix = torch.cumsum(e_end, dim=0)
                prefix_shift = torch.zeros_like(prefix)
                prefix_shift[L_end:, :, :] = prefix[:-L_end, :, :]
                sum_win = prefix - prefix_shift

                # Sliding-window count of valid n-grams (avoid dilution from padded zeros).
                valid_f = valid.to(dtype=torch.float32)
                prefix_c = torch.cumsum(valid_f, dim=0)
                prefix_c_shift = torch.zeros_like(prefix_c)
                prefix_c_shift[L_end:, :] = prefix_c[:-L_end, :]
                cnt_win = prefix_c - prefix_c_shift
                denom = cnt_win.clamp_min(1.0).to(dtype=e_end.dtype).unsqueeze(-1)
                e_win = sum_win / denom

            feats.append(e_win)

        engram_feat = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

        key = self.key_proj(engram_feat)
        value = self.value_proj(engram_feat)

        qn = self.norm_q(hidden_states)
        kn = self.norm_k(key)

        # Gate like demo: dot -> signed sqrt -> sigmoid.
        gate = (qn * kn).sum(dim=-1) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)

        out = gate * value
        if self.short_conv is not None:
            out = out + self.short_conv(out)

        return out
