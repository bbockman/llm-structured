pip install tokenizers

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFD, StripAccents, Lowercase
import glob

# -----------------------------
# 1. Gather text files
# -----------------------------
# You can save your Stage-1 Arrow dataset text to a single .txt file or split into chunks
# Example: concatenate all 'text' fields into corpus.txt
# corpus.txt can be very large; tokenizers handle it efficiently

corpus_files = ["/mnt/xd/ml/hf/datasets/synth_stage1_formatted/data-*.txt"]

# -----------------------------
# 2. Create BPE tokenizer
# -----------------------------
tokenizer = Tokenizer(models.BPE())

# Normalization and pre-tokenization
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = ByteLevel()

# Trainer with special tokens
special_tokens = ["<pad>", "<bos>", "<eos>", "<sep>"] + [f"<lang:{x}>" for x in ["en","es","ja","fr","zh","de"]]

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=special_tokens,
    initial_alphabet=ByteLevel.alphabet(),
)

# -----------------------------
# 3. Train
# -----------------------------
tokenizer.train(files=corpus_files, trainer=trainer)

# -----------------------------
# 4. Save tokenizer for later
# -----------------------------
tokenizer.save("synth_tokenizer.json")



#######################################
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
import datasets

# Load your data
ds = datasets.load_from_disk("/mnt/xd/ml/hf/datasets/synth_stage1_formatted/")

# Extract text iterator
def batch_iterator(batch_size=1000):
    for i in range(0, len(ds), batch_size):
        yield ds[i:i+batch_size]["text"]

# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# Train with your target vocab size
trainer = BpeTrainer(
    vocab_size=32000,  # Your target
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"] + 
                   [f"<lang:{x}>" for x in ["en","es","ja","fr","zh","de"]],
    show_progress=True,
)

print("Training tokenizer on your data...")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# Add post-processor for proper token handling
tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>")),
    ],
)

# Wrap in HuggingFace format
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
)

# Save
fast_tokenizer.save_pretrained("./tokenizers/custom_32k")
print(f"Saved tokenizer with vocab size: {len(fast_tokenizer)}")

# Test it
sample = "This is a test sentence."
encoded = fast_tokenizer.encode(sample)
print(f"\nTest encoding: {encoded}")
print(f"Decoded: {fast_tokenizer.decode(encoded)}")



###############################################
# Replace the tokenizer section:

from transformers import AutoTokenizer

# Option 2a: GPT-2 but trim vocab (quick hack)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Keep only most frequent 32k tokens (this is hacky but works)
vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:32000]
# This won't actually reduce the model embedding size, just limits what you use

# Option 2b: Use a model with naturally smaller vocab
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 30k vocab
# or
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")  # 50k, but can train custom

# Option 2c: Load your custom trained one (after running script above)
tokenizer = AutoTokenizer.from_pretrained("./tokenizers/custom_32k")