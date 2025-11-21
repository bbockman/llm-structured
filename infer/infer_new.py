import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from llm.model import TinyDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": [
        "<sep>"
    ]
})

print(f"Tokenizer vocab size: {len(tokenizer)}")

# Load model
model = TinyDecoder(
    vocab_size=len(tokenizer),
    d_model=384,
    n_layers=8,
    n_heads=8,
    d_ff=1536,
    max_seq=1024
).to(device)

checkpoint = torch.load("tinydecoder_lm_best.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully!")

# ============================================================
# GENERATION FUNCTION WITH DIAGNOSTICS
# ============================================================
@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    show_probs=False
):
    """Generate text from a prompt"""
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    start = input_ids.shape[1]
    for tok in input_ids[0]:
        print(f"  Token ID: {tok.item()} -> '{tokenizer.decode([tok.item()], skip_special_tokens=False)}'")
        
    print(f"\nInput shape: {input_ids.shape}")
    print(f"\nGenerating from prompt (length {start} tokens)...")
    print(f"\nPrompt: {prompt}")
    print(f"Prompt token IDs: {input_ids[0].tolist()}\n")
    
    generated_tokens = []
    
    for step in range(2):
        #logits = model(input_ids)
        logits = model.compute_loss(input_ids)  # For diagnostics
        print(f"\nComputed logits shape: {logits.shape}")
        argmax_ids = torch.argmax(logits[0], dim=-1)  # shape: [seq_len]
        decoded_tokens = tokenizer.decode(argmax_ids.tolist(), skip_special_tokens=False)
        for tok in argmax_ids:
            print(f"  Token ID: {tok.item()} -> '{tokenizer.decode([tok.item()], skip_special_tokens=False)}'") 
        print(f"Step {step}: Argmax tokens for all positions so far:")
        print(f"  Token IDs: {argmax_ids.tolist()}")
        print(f"  Decoded: {decoded_tokens}")
        next_token_logits = logits[0, -1, :] / temperature
        pred_token_id = torch.argmax(next_token_logits).item()
        pred_token_str = tokenizer.decode([pred_token_id])
        print(f"Step {step}: Predicted token (argmax): '{pred_token_str}' (id: {pred_token_id})")

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
            break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
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
    "<bos> Translation regulation in plants operates through multiple coordinated mechanisms including initiation",
    max_new_tokens=10,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    show_probs=True
)
print(f"\nFull output: {output}")



# ============================================================
# REGULAR TESTS
# ============================================================
# test_prompts = [
#     "Once upon a time",
#     "The meaning of life is",
# ]

# print("\n" + "="*80)
# print("TESTING GENERATION")
# print("="*80)

# for prompt in test_prompts:
#     print("\n" + "-"*80)
#     output = generate(prompt, max_new_tokens=50, temperature=0.8, top_k=50)
#     print(f"\nGenerated:\n{output}")

# print("\nDone!")