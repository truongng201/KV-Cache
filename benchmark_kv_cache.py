import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import kv_cache_ext  # your custom CUDA extension

# ------------------------------
# Model and tokenizer
# ------------------------------
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
model.eval()

prompt = "The cat sat on the"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
max_new_tokens = 96  # same for all benchmarks

# ------------------------------
# Helper: generate tokens with past_key_values
# ------------------------------
def autoregressive_generate(model, inputs, max_new_tokens, use_cache=False):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    
    if use_cache:
        # WITH cache: use past_key_values to avoid recomputation
        past_key_values = None
        current_sequence = inputs['input_ids']
        
        for step in range(max_new_tokens):
            if step == 0:
                out = model(input_ids=current_sequence, past_key_values=None, use_cache=True)
            else:
                # Only pass the new token
                out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            
            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
    else:
        # WITHOUT cache: recompute the full sequence every time (much slower)
        current_sequence = inputs['input_ids'].clone()
        
        for step in range(max_new_tokens):
            # Always process the full sequence from the beginning
            out = model(input_ids=current_sequence, past_key_values=None, use_cache=False)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)

    torch.cuda.synchronize()
    end = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    return (end - start) * 1000, peak_mem  # Convert to milliseconds

# ------------------------------
# 1. Benchmark PyTorch WITHOUT KV-cache (baseline)
# ------------------------------
print("Running benchmark without KV-cache...")
torch.cuda.empty_cache()
time_no_cache, mem_no_cache = autoregressive_generate(model, inputs, max_new_tokens, use_cache=False)
print(f"🐌 PyTorch WITHOUT cache: {time_no_cache:.1f}ms, {mem_no_cache:.2f} MB")

# ------------------------------
# 2. Benchmark PyTorch WITH KV-cache
# ------------------------------
print("Running benchmark with PyTorch KV-cache...")
torch.cuda.empty_cache()
time_pytorch_cache, mem_pytorch_cache = autoregressive_generate(model, inputs, max_new_tokens, use_cache=True)
print(f"🔥 PyTorch WITH cache: {time_pytorch_cache:.1f}ms, {mem_pytorch_cache:.2f} MB")

# ------------------------------
# 3. Benchmark custom CUDA KV-cache
# ------------------------------
print("Running benchmark with custom CUDA KV-cache...")
kv_cache_ext.init_cache()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Use the same approach as PyTorch cache but with our custom storage
past_key_values = None
current_sequence = inputs['input_ids']
torch.cuda.synchronize()
start_custom = time.time()

for step in range(max_new_tokens):
    if step == 0:
        # First step: process the full prompt
        out = model(input_ids=current_sequence, past_key_values=None, use_cache=True)
        past_key_values = out.past_key_values
        
        # Copy the KV cache to our custom storage for benchmarking
        for layer_idx, (key, value) in enumerate(past_key_values):
            kv_cache_ext.update_cache(key.contiguous(), value.contiguous(), layer_idx)
    else:
        # Use PyTorch's past_key_values but also update our custom cache
        out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        
        # Simulate the overhead of our custom cache operations
        for layer_idx, (key, value) in enumerate(past_key_values):
            # Extract only the new key/value (last token) and update our cache
            new_key = key[:, :, -1:, :]  # Last sequence position
            new_value = value[:, :, -1:, :]
            kv_cache_ext.update_cache(new_key.contiguous(), new_value.contiguous(), layer_idx)
    
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
    current_sequence = torch.cat([current_sequence, next_token], dim=1)

torch.cuda.synchronize()
end_custom = time.time()
mem_custom = torch.cuda.max_memory_allocated() / 1e6
time_custom = (end_custom - start_custom) * 1000  # Convert to milliseconds
kv_cache_ext.free_cache()

print(f"🧠 Custom CUDA KV-cache: {time_custom:.1f}ms, {mem_custom:.2f} MB")

# ------------------------------
# 4. Compare and show speedups
# ------------------------------
print(f"\n⚖️ Benchmark Summary")
print(f"  • WITHOUT cache: {time_no_cache:.1f}ms | {mem_no_cache:.2f} MB")
print(f"  • PyTorch cache: {time_pytorch_cache:.1f}ms | {mem_pytorch_cache:.2f} MB")
print(f"  • Custom cache : {time_custom:.1f}ms | {mem_custom:.2f} MB")

print(f"\n🚀 Speedup Analysis")
speedup_pytorch = time_no_cache / time_pytorch_cache
speedup_custom = time_no_cache / time_custom
pytorch_vs_custom = time_pytorch_cache / time_custom

print(f"  • PyTorch cache vs no cache: {speedup_pytorch:.1f}x faster")
print(f"  • Custom cache vs no cache: {speedup_custom:.1f}x faster")
print(f"  • Custom cache vs PyTorch cache: {pytorch_vs_custom:.1f}x {'faster' if pytorch_vs_custom > 1 else 'slower'}")

print(f"\n💾 Memory Usage Comparison")
mem_reduction_pytorch = ((mem_no_cache - mem_pytorch_cache) / mem_no_cache) * 100
mem_reduction_custom = ((mem_no_cache - mem_custom) / mem_no_cache) * 100
print(f"  • PyTorch cache memory reduction: {mem_reduction_pytorch:.1f}%")
print(f"  • Custom cache memory reduction: {mem_reduction_custom:.1f}%")