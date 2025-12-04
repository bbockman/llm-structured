import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tokenizer.pleias_tok import PleiasTokenizer
tokenizer = PleiasTokenizer().base

print(f"Tokenizer vocab size: {len(tokenizer)}")

from llm.model_flash import get_current_model
model = get_current_model(vocab_size=len(tokenizer)).to(device)

checkpoint = torch.load("disk/models/llm-scoped/params_134autocast.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("Model loaded successfully!")

@torch.no_grad()
def generate(prompt):
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    generated_tokens = []
    
    max_new_tok = 1024 - input_ids.shape[1]

    while max_new_tok > 0:
        max_new_tok -= 1
        # logits = model(input_ids)
        # next_token = torch.argmax(logits[0, -1, :]).view(1, 1)
        # generated_tokens.append(next_token.item())
        # input_ids = torch.cat([input_ids, next_token], dim=1)

        # simple top-k sampling: k=20
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        k = 20
        topk_vals, topk_idx = torch.topk(next_token_logits, k)
        probs = torch.softmax(topk_vals, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        next_token = topk_idx[sampled_idx].view(1, 1)
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            generated_tokens.append(tokenizer.eos_token_id)
            break
    
    print("\nTotal generated tokens:", len(generated_tokens))
    if generated_tokens[-1] != tokenizer.eos_token_id:
        print("Warning: Generation stopped before EOS token was produced.")

    return tokenizer.decode(input_ids[0], skip_special_tokens=False)


print("\n" + "="*80)
print("DIAGNOSTIC TEST - Showing training data predictions")
print("="*80)

from data.load_shard import load_synth_shards
ds = load_synth_shards(num_shards=1, total_shards=10, pattern_prefix="data",
    base_path="/mnt/xd/ml/hf/datasets/synth_stage1_formatted/"
)

for i, example in enumerate(ds):
    if i >= 10: 
        break            
    decoded_prompt = tokenizer.decode(example["input_ids"][:20], skip_special_tokens=False)
    print("\n--- Generating for prompt ---")
    print(decoded_prompt)
    generated_text = generate(decoded_prompt)
    print("\n--- Generated Text ---")
    print(generated_text)
    print("\n -- Full Example Target Text -- ")
    print(tokenizer.decode(example["input_ids"], skip_special_tokens=False))

print("\nDone!")
