import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tokenizer.pleias_tok import PleiasTokenizer
tokenizer = PleiasTokenizer().base

print(f"Tokenizer vocab size: {len(tokenizer)}")

# Load model
from llm.model_flash import get_current_model
model = get_current_model(vocab_size=len(tokenizer)).to(device)

checkpoint = torch.load("disk/models/llm-scoped/params_snap.pth", map_location=device)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

print("Model loaded successfully!")

# ============================================================
# GENERATION FUNCTION WITH DIAGNOSTICS
# ============================================================
@torch.no_grad()
def generate(
    prompt,
    max_tokens=1024,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    show_probs=False
):
    """Generate text from a prompt"""
    
    input_ids = tokenizer.encode(prompt, 
                                 return_tensors="pt",
                                 add_special_tokens=False).to(device)
    print(f"Input IDs: {input_ids}")
    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {input_ids.shape[1]}")
    
    generated_tokens = []
    
    for step in range(max_tokens-input_ids.shape[1]):
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :] / temperature
        
        # Show top predictions
        if show_probs and step < 5:
            top_probs, top_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), k=10)
            print(f"\nStep {step} - Top 10 predictions:")
            for prob, idx in zip(top_probs, top_indices):
                token = tokenizer.decode([idx.item()])
                print(f"  '{token}': {prob.item():.4f}")
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            generated_tokens.append(tokenizer.eos_token_id)
            break
    
    print("\nTotal generated tokens:", len(generated_tokens))
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    
    if show_probs:
        print("\nGenerated tokens:")
        for i, tok_id in enumerate(generated_tokens[:10]):
            print(f"  {i}: '{tokenizer.decode([tok_id])}' (id: {tok_id})")
    
    return generated_text

# ============================================================
# DIAGNOSTIC TEST FIRST
# ============================================================
print("\n" + "="*80)
print("DIAGNOSTIC TEST - Showing what model predicts")
print("="*80)

output = generate(
    f"{tokenizer.bos_token} Translation regulation in plants operates through multiple coordinated mechanisms including initiation",
    max_tokens=1024,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    show_probs=True
)
print(f"\nFull output: {output}")

# ============================================================
# REGULAR TESTS
# ============================================================
test_prompts = [
    f"{tokenizer.bos_token} Artificial intelligence is",
    f"{tokenizer.bos_token} Geometric series converge when",
    f"{tokenizer.bos_token} In a distant future, humanity has",
    f"{tokenizer.bos_token} During World War II,",
    f"{tokenizer.bos_token} The Deep South's"
]

print("\n" + "="*80)
print("TESTING GENERATION")
print("="*80)

for prompt in test_prompts:
    print("\n" + "-"*80)
    output = generate(prompt)
    print(f"\nGenerated:\n{output}")

print("\nDone!")