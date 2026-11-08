"""Build OFFLINE fixtures to test the pipeline without internet:
  * tests/fixtures/tiny_bert      : randomly initialised miniature BERT + tokenizer
  * tests/fixtures/tiny_sbert     : miniature SentenceTransformer built from tiny_bert
  * tests/fixtures/synthetic.csv  : synthetic TAWOS-like data (6 projects)
  * tests/fixtures/pairs.csv      : 3 pairs among those projects
The models are random, so numbers from them mean nothing; they only prove that the
real 'hf' code path (AutoTokenizer/AutoModel/SentenceTransformer) runs end to end.
"""
import os
import random
import sys

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(OUT, exist_ok=True)
WORDS = ("fix login error user password page crash button api timeout database query slow memory leak "
         "add feature support export report dashboard upgrade library test failing build deploy config "
         "null pointer exception docs update ui layout broken mobile android ios plugin").split()


def make_models():
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from sentence_transformers import SentenceTransformer, models

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS)) + [".", ","]
    d = os.path.join(OUT, "tiny_bert")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))
    BertTokenizerFast(os.path.join(d, "vocab.txt"), do_lower_case=True).save_pretrained(d)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=48, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=256)
    BertModel(cfg).save_pretrained(d)
    word = models.Transformer(d, max_seq_length=64)
    pool = models.Pooling(word.get_word_embedding_dimension(), "mean")
    SentenceTransformer(modules=[word, pool]).save(os.path.join(OUT, "tiny_sbert"))


def make_data():
    rnd = random.Random(0)
    rows, iid = [], 0
    sizes = {"Alpha Project": 120, "Beta Project": 90, "Gamma Project": 150, "Delta Project": 80,
             "Epsilon Project": 110, "Tiny Project": 20}
    for proj, n in sizes.items():
        base = pd.Timestamp("2018-01-01")
        for i in range(n):
            words = [rnd.choice(WORDS) for _ in range(rnd.randint(5, 40))]
            sp = int(np.clip(round(len(words) / 8 + rnd.choice([0, 1, 1, 2, 3])), 1, 13))
            rows.append({"Issue_ID": iid, "Project_Name": proj, "Title": " ".join(words[:5]),
                         "Description_Text": " ".join(words[5:]) + " http://x.y/z {code}ignored{code}",
                         "Creation_Date": (base + pd.Timedelta(days=i)).isoformat(), "Story_Point": sp})
            iid += 1
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "synthetic.csv"), index=False)
    pd.DataFrame({"pair_id": [1, 2, 3],
                  "train_project": ["Alpha", "Beta Project", "Tiny"],
                  "test_project": ["Beta", "Gamma", "Delta"]}).to_csv(os.path.join(OUT, "pairs.csv"), index=False)


if __name__ == "__main__":
    make_data()
    make_models()
    print("fixtures written to", OUT)
