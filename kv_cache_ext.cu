#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <unordered_map>

// More efficient cache structure that stores layer-wise KV pairs
struct LayerKVCache {
    torch::Tensor key_cache;
    torch::Tensor value_cache;
    int current_length;
    int max_length;
    bool initialized;
    
    LayerKVCache() : current_length(0), max_length(0), initialized(false) {}
};

// Global cache storage - one cache per layer
std::unordered_map<int, LayerKVCache> layer_caches;
int max_layers = 32; // Reasonable default for most models

void init_cache() {
    layer_caches.clear();
}

// Optimized update that uses tensor views instead of memory copying
std::pair<torch::Tensor, torch::Tensor> update_cache(torch::Tensor key, torch::Tensor value, int layer_idx) {
    auto& cache = layer_caches[layer_idx];
    
    // Get dimensions
    auto key_shape = key.sizes().vec();
    auto value_shape = value.sizes().vec();
    int batch_size = key_shape[0];
    int num_heads = key_shape[1];
    int new_seq_len = key_shape[2];
    int head_dim = key_shape[3];
    
    if (!cache.initialized) {
        // First time - allocate with reasonable buffer size
        int buffer_size = std::max(512, new_seq_len * 4); // Buffer for future tokens
        
        cache.key_cache = torch::zeros({batch_size, num_heads, buffer_size, head_dim}, 
                                       torch::TensorOptions().device(key.device()).dtype(key.dtype()));
        cache.value_cache = torch::zeros({batch_size, num_heads, buffer_size, head_dim}, 
                                         torch::TensorOptions().device(value.device()).dtype(value.dtype()));
        cache.max_length = buffer_size;
        cache.current_length = 0;
        cache.initialized = true;
    }
    
    // Check if we need to expand the cache
    if (cache.current_length + new_seq_len > cache.max_length) {
        int new_max = std::max(cache.max_length * 2, cache.current_length + new_seq_len);
        
        // Expand cache tensors
        auto new_key_cache = torch::zeros({batch_size, num_heads, new_max, head_dim}, 
                                          torch::TensorOptions().device(key.device()).dtype(key.dtype()));
        auto new_value_cache = torch::zeros({batch_size, num_heads, new_max, head_dim}, 
                                            torch::TensorOptions().device(value.device()).dtype(value.dtype()));
        
        // Copy existing data efficiently using tensor slicing
        if (cache.current_length > 0) {
            new_key_cache.slice(2, 0, cache.current_length).copy_(
                cache.key_cache.slice(2, 0, cache.current_length));
            new_value_cache.slice(2, 0, cache.current_length).copy_(
                cache.value_cache.slice(2, 0, cache.current_length));
        }
        
        cache.key_cache = new_key_cache;
        cache.value_cache = new_value_cache;
        cache.max_length = new_max;
    }
    
    // Append new key-value pairs using efficient tensor slicing
    cache.key_cache.slice(2, cache.current_length, cache.current_length + new_seq_len).copy_(key);
    cache.value_cache.slice(2, cache.current_length, cache.current_length + new_seq_len).copy_(value);
    
    cache.current_length += new_seq_len;
    
    // Return views of the full cached sequences
    auto cached_key = cache.key_cache.slice(2, 0, cache.current_length);
    auto cached_value = cache.value_cache.slice(2, 0, cache.current_length);
    
    return std::make_pair(cached_key, cached_value);
}

void free_cache() {
    layer_caches.clear();
}

// Get cached KV for a specific layer
std::pair<torch::Tensor, torch::Tensor> get_cached_kv(int layer_idx) {
    auto it = layer_caches.find(layer_idx);
    if (it != layer_caches.end() && it->second.initialized) {
        auto& cache = it->second;
        auto cached_key = cache.key_cache.slice(2, 0, cache.current_length);
        auto cached_value = cache.value_cache.slice(2, 0, cache.current_length);
        return std::make_pair(cached_key, cached_value);
    }
    // Return empty tensors if not found
    return std::make_pair(torch::Tensor(), torch::Tensor());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_cache", &init_cache, "Initialize the KV cache");
    m.def("update_cache", &update_cache, "Update cache with new key-value pairs", 
          py::arg("key"), py::arg("value"), py::arg("layer_idx"));
    m.def("get_cached_kv", &get_cached_kv, "Get cached KV for a layer", 
          py::arg("layer_idx"));
    m.def("free_cache", &free_cache, "Free all cached memory");
}

