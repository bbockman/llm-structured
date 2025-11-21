import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from llm.model import TinyDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_langs = [f"<lang:{x}>" for x in ["en","es","ja","fr","zh","de"]]
tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<sep>", *special_langs]
})

# Load model
model = TinyDecoder(
    vocab_size=len(tokenizer),
    d_model=384,
    n_layers=8,  # ✅ 8 layers now
    n_heads=8,
    d_ff=1536,
    max_seq=2048
).to(device)

checkpoint = torch.load("tinydecoder_lm_best.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

print("Model loaded! (31M params, 8 layers)\n")

@torch.no_grad()
def generate(prompt, max_tokens=50, temp=0.8):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temp
        
        # Simple top-k sampling
        top_k = 50
        indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
        next_logits[indices_to_remove] = float('-inf')
        
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Quick tests
tests = [
    "Once upon a time",
    "The quick brown fox",
    "In the beginning",
    "def factorial(n):",
    "Question: What is 2+2? Answer:",
]

print("="*80)
print("QUICK GENERATION TEST (8 layers, 1 epoch)")
print("="*80)

for prompt in tests:
    print(f"\n📝 Prompt: {prompt}")
    output = generate(prompt, max_tokens=30, temp=0.8)
    print(f"🤖 Output: {output}")
    print("-"*80)

print("\n✅ Done! How does it look?")