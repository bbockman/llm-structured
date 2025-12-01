#!/usr/bin/env python3
"""
Train a SentencePiece Unigram tokenizer directly from Arrow shards
using Dataset.from_file(), with shard subset control,
consistent normalization, custom tokens, and **industrial-scale sampling**.
"""

import re
import random
import sentencepiece as spm
from datasets import Dataset
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------

ARROW_DIR = "/mnt/xd/ml/hf/datasets/PleIAs___synth/default/0.0.0/6ebe6a97043747aa5f2232ea1182841c4a6afcb0"
VOCAB_SIZE = 8192
SHARD_INDICES = slice(0, 20)

# Sample probability (industrial tokenizer training)
# ~ 50M lines total is typical for LLM-scale tokenizers
SAMPLE_PROB = 0.02    # about 2% → tune based on shard size

# Custom tokens
SPECIAL_TOKENS = [
    "<par>", "<sys>", "<que>", "<ans>", "<cot>", "<rag>", "<new>"
]

# Use SentencePiece built-ins:
# <unk> = 0, <s> = 1, </s> = 2

# -----------------------------
# NORMALIZATION
# -----------------------------

PARA_RE = re.compile(r"\n{2,}")

def normalize_paragraphs(text):
    text = PARA_RE.sub(" <par> ", text.strip())
    return text.replace("\n", " <new> ").strip()

# -----------------------------
# SHARD LOADING
# -----------------------------

def list_shard_paths():
    files = sorted(Path(ARROW_DIR).glob("*.arrow"))
    if not files:
        raise RuntimeError(f"No arrow files found in {ARROW_DIR}")
    return files

def load_shard_subset(shard_indices):
    all_files = list_shard_paths()
    shard_files = all_files[shard_indices]

    print(f"Using {len(shard_files)} shard(s):")
    for f in shard_files:
        print("   ", f.name)

    return [Dataset.from_file(str(p)) for p in shard_files]

# -----------------------------
# SENTENCE ITERATOR WITH SAMPLING
# -----------------------------

def sentence_iterator(shard_indices):
    """
    Industrial-grade iterator:
    - Loads selected Arrow shards (memory mapped)
    - Normalizes data
    - Samples at rate SAMPLE_PROB to avoid enormous corpora
    - Uses SP defaults (<unk>, <s>, </s>)
    - Adds custom tokens explicitly
    """

    datasets = load_shard_subset(shard_indices)
    total_seen = 0
    total_yielded = 0

    for ds in datasets:
        for ex in ds:
            total_seen += 1

            # Language filter
            if ex.get("language") != "en":
                continue

            # Sampling (industrial standard)
            if random.random() > SAMPLE_PROB:
                continue

            text = ex.get("synthetic_answer")
            if not text:
                continue

            text = normalize_paragraphs(text)

            if text:
                total_yielded += 1
                if total_yielded % 100000 == 0:
                    print(f"  yielded {total_yielded:,} samples...")
                yield text

    print(f"\nSeen {total_seen:,} examples, yielded {total_yielded:,} samples.")

# -----------------------------
# TRAIN TOKENIZER
# -----------------------------

def main():
    print("Preparing SentencePiece tokenizer...")

    user_syms = ",".join(SPECIAL_TOKENS)

    spm.SentencePieceTrainer.Train(
        sentence_iterator=sentence_iterator(SHARD_INDICES),
        model_prefix="unigram_8k",
        model_type="unigram",
        vocab_size=VOCAB_SIZE,

        # Built-in special tokens
        unk_id=0,            # <unk>
        bos_id=1,            # <s>
        eos_id=2,            # </s>
        pad_id=3,            # <pad>

        character_coverage=1.0,
        user_defined_symbols=user_syms,

        input_sentence_size=50_000_000,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1_000_000,
        max_sentence_length=1024,
        train_extremely_large_corpus=True,
        num_threads=16,
    )

    print("\nDone. Tokenizer written to unigram_8k.model")

if __name__ == "__main__":
    main()
