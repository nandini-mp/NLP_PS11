from __future__ import annotations

import hashlib
import os
import re

import numpy as np


def _cache_key(cfg: dict, project: str, ids) -> str:
    e = cfg["embeddings"]
    blob = "|".join(
        [project, e["backend"], e["bert_model"], str(e["bert_layer"]), e["sbert_model"],
         str(cfg["text"]["max_tokens"]), str(e.get("hash_dim", "")),
         hashlib.md5(",".join(map(str, ids)).encode()).hexdigest()]
    )
    return hashlib.md5(blob.encode()).hexdigest()[:16]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def _hash_vec(token: str, dim: int) -> np.ndarray:
    seed = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    return np.random.RandomState(seed).randn(dim).astype(np.float32)


def _hash_backend(texts, max_len, bert_dim, sbert_dim):
    n = len(texts)
    tok = np.zeros((n, max_len, bert_dim), np.float16)
    mask = np.zeros((n, max_len), np.uint8)
    sent = np.zeros((n, sbert_dim), np.float32)
    for i, t in enumerate(texts):
        words = t.lower().split()[: max_len - 1]
        vecs = [_hash_vec("[CLS]", bert_dim)] + [_hash_vec(w, bert_dim) for w in words]
        tok[i, : len(vecs)] = np.stack(vecs)
        mask[i, : len(vecs)] = 1
        sent[i] = np.mean([_hash_vec(w, sbert_dim) for w in words] or [np.zeros(sbert_dim)], axis=0)
    return tok, mask, sent


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def bert_token_embeddings(texts, model_name, layer, max_len, batch_size):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = _device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(dev).eval()
    hidden = model.config.hidden_size
    n = len(texts)
    tok = np.zeros((n, max_len, hidden), np.float16)
    mask = np.zeros((n, max_len), np.uint8)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            batch = texts[s : s + batch_size]
            enc = tokenizer(batch, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(dev)
            out = model(**enc).hidden_states[layer]
            tok[s : s + len(batch)] = out.cpu().numpy().astype(np.float16)
            mask[s : s + len(batch)] = enc["attention_mask"].cpu().numpy().astype(np.uint8)
    return tok, mask


def sbert_embeddings(texts, model_name, batch_size):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=_device())
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                        convert_to_numpy=True).astype(np.float32)


def get_embeddings(cfg: dict, project: str, ids, texts):
    e = cfg["embeddings"]
    max_len = cfg["text"]["max_tokens"]
    mode = cfg["model"]["mode"]
    need_bert = mode in ("bert", "combined")    # skip what the chosen representation does not use
    need_sbert = mode in ("sbert", "combined")
    cdir = os.path.join(e["cache_dir"], f"{_slug(project)}_{_cache_key(cfg, project, ids)}_{int(need_bert)}{int(need_sbert)}")
    files = {k: os.path.join(cdir, f"{k}.npy") for k in ("tok", "mask", "sent")}
    if not all(os.path.exists(f) for f in files.values()):
        os.makedirs(cdir, exist_ok=True)
        texts = list(texts)
        n = len(texts)
        if e["backend"] == "hash":
            d = e.get("hash_dim", 32)
            tok, mask, sent = _hash_backend(texts, max_len, d, max(d // 2, 8))
        elif e["backend"] == "hf":
            if need_bert:
                tok, mask = bert_token_embeddings(texts, e["bert_model"], e["bert_layer"],
                                                  max_len, e["batch_size"])
            if need_sbert:
                sent = sbert_embeddings(texts, e["sbert_model"], e["batch_size"])
        else:
            raise ValueError(f"unknown embeddings.backend {e['backend']}")
        if not need_bert:
            tok, mask = np.zeros((n, 1, 1), np.float16), np.ones((n, 1), np.uint8)
        if not need_sbert:
            sent = np.zeros((n, 1), np.float32)
        for k, arr in zip(("tok", "mask", "sent"), (tok, mask, sent)):
            np.save(files[k], arr)
    return (np.load(files["tok"], mmap_mode="r"), np.load(files["mask"]), np.load(files["sent"]))
