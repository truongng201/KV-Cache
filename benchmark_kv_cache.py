import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import kv_cache_ext

model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
model.eval()

prompt = "The cat sat on the"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
max_new_tokens = 96  # same for all benchmarks

def autoregressive_generate(model, inputs, max_new_tokens, use_cache=False):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    
    if use_cache:
        past_key_values = None
        current_sequence = inputs['input_ids']
        
        for step in range(max_new_tokens):
            if step == 0:
                out = model(input_ids=current_sequence, past_key_values=None, use_cache=True)
            else:
                # Only pass thenew token
                out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            
            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
    else:
        current_sequence = inputs['input_ids'].clone()
        
        for step in range(max_new_tokens):
            out = model(input_ids=current_sequence, past_key_values=None, use_cache=False)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)

    torch.cuda.synchronize()
    end = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    return (end - start) * 1000, peak_mem  # Convert to milliseconds

torch.cuda.empty_cache()
time_no_cache, mem_no_cache = autoregressive_generate(model, inputs, max_new_tokens, use_cache=False)


torch.cuda.empty_cache()
time_pytorch_cache, mem_pytorch_cache = autoregressive_generate(model, inputs, max_new_tokens, use_cache=True)


kv_cache_ext.init_cache()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

past_key_values = None
current_sequence = inputs['input_ids']
torch.cuda.synchronize()
start_custom = time.time()

for step in range(max_new_tokens):
    if step == 0:
        out = model(input_ids=current_sequence, past_key_values=None, use_cache=True)
        past_key_values = out.past_key_values
        
        for layer_idx, (key, value) in enumerate(past_key_values):
            kv_cache_ext.update_cache(key.contiguous(), value.contiguous(), layer_idx)
    else:
        out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        
        for layer_idx, (key, value) in enumerate(past_key_values):
            new_key = key[:, :, -1:, :]
            new_value = value[:, :, -1:, :]
            kv_cache_ext.update_cache(new_key.contiguous(), new_value.contiguous(), layer_idx)
    
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
    current_sequence = torch.cat([current_sequence, next_token], dim=1)

torch.cuda.synchronize()
end_custom = time.time()
mem_custom = torch.cuda.max_memory_allocated() / 1e6
time_custom = (end_custom - start_custom) * 1000
kv_cache_ext.free_cache()

print(f"Benchmark Summary")
print(f"  • WITHOUT cache: {time_no_cache:.1f}ms | {mem_no_cache:.2f} MB")
print(f"  • PyTorch cache: {time_pytorch_cache:.1f}ms | {mem_pytorch_cache:.2f} MB")
print(f"  • Custom cache : {time_custom:.1f}ms | {mem_custom:.2f} MB")

print(f"\nSpeedup Analysis")
speedup_pytorch = time_no_cache / time_pytorch_cache
speedup_custom = time_no_cache / time_custom
pytorch_vs_custom = time_pytorch_cache / time_custom

print(f"  • PyTorch cache vs no cache: {speedup_pytorch:.1f}x faster")
print(f"  • Custom cache vs no cache: {speedup_custom:.1f}x faster")
print(f"  • Custom cache vs PyTorch cache: {pytorch_vs_custom:.1f}x {'faster' if pytorch_vs_custom > 1 else 'slower'}")

print(f"\nMemory Usage Comparison")
mem_reduction_pytorch = ((mem_no_cache - mem_pytorch_cache) / mem_no_cache) * 100
mem_reduction_custom = ((mem_no_cache - mem_custom) / mem_no_cache) * 100
print(f"  • PyTorch cache memory reduction: {mem_reduction_pytorch:.1f}%")
print(f"  • Custom cache memory reduction: {mem_reduction_custom:.1f}%")