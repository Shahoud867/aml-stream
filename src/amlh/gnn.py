"""Direction-aware two-hop temporal GNN over sampled transaction neighbourhoods.

Message passing is separated by direction (in/out), as in reverse message passing and the Phase 2
design. Pooling uses mean, max and log-count so aggregation stays sensitive to cardinality.
The same architecture runs at every horizon: slots a horizon cannot fill stay padded.

Ablations: shared_direction (A6, one parameter set for both directions), one_hop (A7, no second hop),
no_count (A8, the count term of the pooling is removed).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def masked_pool(x: torch.Tensor, mask: torch.Tensor, use_count: bool = True) -> torch.Tensor:
    """x[..., n, d], mask[..., n] -> [..., 2d + 1]: mean, max, log1p(count)."""
    m = mask.unsqueeze(-1).to(x.dtype)
    cnt = m.sum(-2)
    mean = (x * m).sum(-2) / cnt.clamp(min=1.0)
    mx = x.masked_fill(~mask.unsqueeze(-1), -1e4).max(-2).values
    mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
    count = torch.log1p(cnt) if use_count else torch.zeros_like(cnt)
    return torch.cat([mean, mx, count], -1)


class EdgeEncoder(nn.Module):
    def __init__(self, n_cur: int, n_fmt: int, d: int):
        super().__init__()
        self.cur_p = nn.Embedding(n_cur + 1, 6)
        self.cur_r = nn.Embedding(n_cur + 1, 6)
        self.fmt = nn.Embedding(n_fmt + 1, 4)
        self.proj = nn.Sequential(nn.Linear(6 + 6 + 4 + 5, d), nn.ReLU())

    def forward(self, cont: torch.Tensor, cur_p: torch.Tensor, cur_r: torch.Tensor, fmt: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([cont, self.cur_p(cur_p), self.cur_r(cur_r), self.fmt(fmt)], -1))


class TemporalDirectedGNN(nn.Module):
    def __init__(self, n_cur: int, n_fmt: int, d: int = 64, dropout: float = 0.1,
                 shared_direction: bool = False, one_hop: bool = False, no_count: bool = False):
        super().__init__()
        self.enc = EdgeEncoder(n_cur, n_fmt, d)
        self.phi2 = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.one_hop = one_hop
        self.use_count = not no_count
        self.pool_dim = 2 * d + 1
        pool = self.pool_dim
        make_phi = lambda: nn.Sequential(nn.Linear(d + 2 * pool, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())  # noqa: E731
        first = make_phi()
        self.phi1 = nn.ModuleList([first, first if shared_direction else make_phi()])
        self.head = nn.Sequential(nn.Linear(d + 4 * pool, 2 * d), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, seed_e, hop1_e, hop1_mask, hop2_e, hop2_mask):
        # hop2: [B,2,2,S1,2,S2,d] -> pooled per (first-hop neighbour, direction)
        if self.one_hop:
            shape = hop1_e.shape[:-1] + (2 * self.pool_dim,)
            hx = torch.zeros(shape, dtype=hop1_e.dtype, device=hop1_e.device)
        else:
            z2 = self.phi2(hop2_e)
            p_in = masked_pool(z2[:, :, :, :, 0], hop2_mask[:, :, :, :, 0], self.use_count)
            p_out = masked_pool(z2[:, :, :, :, 1], hop2_mask[:, :, :, :, 1], self.use_count)
            hx = torch.cat([p_in, p_out], -1)                            # [B,2,2,S1,2p]
        msg = torch.cat([hop1_e, hx], -1)                                # [B,2,2,S1,d+2p]
        m_in = self.phi1[0](msg[:, :, 0])                                # incoming neighbours
        m_out = self.phi1[1](msg[:, :, 1])                               # outgoing neighbours
        h_in = masked_pool(m_in, hop1_mask[:, :, 0], self.use_count)     # [B,2,p]
        h_out = masked_pool(m_out, hop1_mask[:, :, 1], self.use_count)
        h = torch.cat([h_in, h_out], -1).flatten(1)                      # [B,4p]
        return self.head(torch.cat([seed_e, h], -1)).squeeze(-1)
