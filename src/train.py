from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import StoryPointNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y_true, y_pred) -> dict:
    err = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
    return {"MAE": float(err.mean()), "MdAE": float(np.median(err))}


class Block:
    def __init__(self, tok, mask, sent, y, idx):
        self.tok, self.mask, self.sent, self.y, self.idx = tok, mask, sent, y, np.asarray(idx)


class _DS(Dataset):
    def __init__(self, b: Block):
        self.b = b

    def __len__(self):
        return len(self.b.idx)

    def __getitem__(self, i):
        j = self.b.idx[i]
        return (torch.from_numpy(np.asarray(self.b.tok[j], dtype=np.float32)),
                torch.from_numpy(self.b.mask[j].astype(np.float32)),
                torch.from_numpy(self.b.sent[j]),
                torch.tensor(self.b.y[j], dtype=torch.float32))


def _loader(b, bs, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(_DS(b), batch_size=bs, shuffle=shuffle, generator=g, num_workers=0)


def predict(model, b: Block, device, bs=64, clip_max=None):
    model.eval()
    out = []
    with torch.no_grad():
        for tok, mask, sent, _ in _loader(b, bs, False, 0):
            out.append(model(tok.to(device), mask.to(device), sent.to(device)).cpu().numpy())
    p = np.concatenate(out)
    return np.clip(p, 0, clip_max) if clip_max is not None else p


def fit(cfg, hp, train: Block, val: Block | None, seed, device, fixed_epochs=None):
    set_seed(seed)
    tc, mc = cfg["train"], cfg["model"]
    y_tr = train.y[train.idx]
    model = StoryPointNet(
        mc["mode"], train.tok.shape[2], train.sent.shape[1], tuple(mc["conv_channels"]),
        mc["kernel_size"], mc["sbert_proj_dim"], mc["head_hidden"], hp["dropout"],
        init_bias=float(np.median(y_tr)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=tc["weight_decay"])
    loss_fn = torch.nn.HuberLoss(delta=tc["huber_delta"])
    clip_max = float(y_tr.max())
    epochs = fixed_epochs if val is None else tc["max_epochs"]
    best, best_state, best_ep, bad = np.inf, None, 1, 0
    for ep in range(1, epochs + 1):
        model.train()
        for tok, mask, sent, y in _loader(train, tc["batch_size"], True, seed + ep):
            opt.zero_grad()
            loss = loss_fn(model(tok.to(device), mask.to(device), sent.to(device)), y.to(device))
            loss.backward()
            opt.step()
        if val is not None:
            mae = metrics(val.y[val.idx], predict(model, val, device, clip_max=clip_max))["MAE"]
            if mae < best - 1e-9:
                best, best_state, best_ep, bad = mae, copy.deepcopy(model.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= tc["patience"]:
                    break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    elif val is None:
        best_ep = epochs
    model.clip_max = clip_max
    return model, best_ep
