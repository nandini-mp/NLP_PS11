from __future__ import annotations

import torch
import torch.nn as nn


class StoryPointNet(nn.Module):
    def __init__(self, mode, bert_dim, sbert_dim, channels=(128, 128, 128), kernel=3,
                 sbert_proj=128, hidden=128, dropout=0.3, init_bias=0.0):
        super().__init__()
        assert mode in ("bert", "sbert", "combined")
        self.mode = mode
        feat = 0
        if mode in ("bert", "combined"):
            layers, cin = [], bert_dim
            for cout in channels:
                layers += [nn.Conv1d(cin, cout, kernel, padding=kernel // 2), nn.ReLU(), nn.Dropout(dropout)]
                cin = cout
            self.cnn = nn.Sequential(*layers)
            feat += cin
        if mode in ("sbert", "combined"):
            self.sproj = nn.Sequential(nn.Linear(sbert_dim, sbert_proj), nn.ReLU(), nn.Dropout(dropout))
            feat += sbert_proj
        self.head = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.head[-1].bias.fill_(float(init_bias))

    def forward(self, tok, mask, sent):
        parts = []
        if self.mode in ("bert", "combined"):
            h = self.cnn(tok.transpose(1, 2))                     
            h = h.masked_fill(mask.unsqueeze(1) == 0, -1e9)       
            parts.append(h.max(dim=2).values)
        if self.mode in ("sbert", "combined"):
            parts.append(self.sproj(sent))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)
