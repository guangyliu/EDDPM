#!/bin/bash
# The tokenized Yelp / Amazon review datasets used by the text experiments are on the
# Hugging Face Hub (fields: bert_token, gpt2_token; splits: train, test).
# The training script loads them directly with `datasets.load_dataset(...)`, so nothing
# needs to be downloaded by hand; this pre-fetches them into the local HF cache.
python - <<'PY'
from datasets import load_dataset
for name in ["guangyil/yelp_short", "guangyil/amazon_tokenized"]:
    ds = load_dataset(name)
    print(name, {k: len(v) for k, v in ds.items()})
PY
