from tokenizers import ByteLevelBPETokenizer

# Prepare a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train on your files
tokenizer.train(
    files=["data1.txt", "data2.txt"],
    vocab_size=151_643,  # same as Qwen
    min_frequency=2,
    special_tokens=[
        "<s>",    # or SOS / BOS
        "</s>",   # EOS
        "<unk>",  # unknown
    ]
)

# Save for reuse
tokenizer.save_model("your_qwen2_tokenizer", prefix="qwen2")
