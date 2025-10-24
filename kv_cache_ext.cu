#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <unordered_map>
#include <memory>

// CUDA kernel for concatenating new key/value data to the cache
__global__ void append_kv_kernel(
    float* dst, const float* src, // Assuming float for simplicity; adjust for float16/bfloat16
    int batch_size, int num_heads, int head_dim, int seq_len, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len * head_dim;
    if (idx < total_elements) {
        int batch = idx / (num_heads * seq_len * head_dim);
        int head = (idx / (seq_len * head_dim)) % num_heads;
        int seq = (idx / head_dim) % seq_len;
        int dim = idx % head_dim;
        int dst_idx = batch * num_heads * (offset + seq_len) * head_dim +
                      head * (offset + seq_len) * head_dim + 
                      (offset + seq) * head_dim + dim;
        int src_idx = batch * num_heads * seq_len * head_dim +
                      head * seq_len * head_dim +
                      seq * head_dim + dim;
        dst[dst_idx] = src[src_idx];
    }
}

struct LayerKVCache {
    void* key_cache_ptr;
    void* value_cache_ptr;
    torch::Tensor key_tensor;
    torch::Tensor value_tensor;
    torch::Tensor key_view; // Cached sliced view
    torch::Tensor value_view; // Cached sliced view
    
    int batch_size;
    int num_heads;
    int head_dim;
    int current_length;
    int max_length;
    int dtype_size;
    
    cudaStream_t cache_stream;
    cudaStream_t copy_stream;
    
    bool initialized;
    
    LayerKVCache() : key_cache_ptr(nullptr), value_cache_ptr(nullptr),
                     current_length(0), max_length(0), initialized(false),
                     cache_stream(nullptr), copy_stream(nullptr) {}
    
    ~LayerKVCache() {
        cleanup();
    }
    
    void cleanup() {
        if (key_cache_ptr) {
            cudaFree(key_cache_ptr);
            key_cache_ptr = nullptr;
        }
        if (value_cache_ptr) {
            cudaFree(value_cache_ptr);
            value_cache_ptr = nullptr;
        }
        if (cache_stream) {
            cudaStreamDestroy(cache_stream);
        }
        if (copy_stream && copy_stream != cache_stream) {
            cudaStreamDestroy(copy_stream);
        }
        initialized = false;
    }
};

static std::unordered_map<int, std::unique_ptr<LayerKVCache>> layer_caches;
static int max_layers = 32;

void init_cache() {
    layer_caches.clear();
    for (int i = 0; i < max_layers; ++i) {
        layer_caches[i] = std::make_unique<LayerKVCache>();
        cudaStreamCreateWithFlags(&layer_caches[i]->cache_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&layer_caches[i]->copy_stream, cudaStreamNonBlocking);
    }
}

int get_dtype_size(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case torch::kFloat32: return 4;
        case torch::kFloat16: return 2;
        case torch::kBFloat16: return 2;
        case torch::kFloat64: return 8;
        default: return 4;
    }
}

std::pair<torch::Tensor, torch::Tensor> update_cache(
    torch::Tensor key, torch::Tensor value, int layer_idx) {
    
    if (layer_caches.find(layer_idx) == layer_caches.end()) {
        throw std::runtime_error("Cache not initialized for layer");
    }
    
    auto& cache = layer_caches[layer_idx];
    
    // Get dimensions
    auto key_shape = key.sizes().vec();
    int batch_size = key_shape[0];
    int num_heads = key_shape[1];
    int new_seq_len = key_shape[2];
    int head_dim = key_shape[3];
    int dtype_size = get_dtype_size(key);
    
    if (!cache->initialized) {
        cache->batch_size = batch_size;
        cache->num_heads = num_heads;
        cache->head_dim = head_dim;
        cache->dtype_size = dtype_size;
        cache->current_length = 0;
        
        // Pre-allocate very large buffer to avoid reallocations
        cache->max_length = std::max(4096, new_seq_len * 32); // Aggressive pre-allocation
        size_t cache_size = batch_size * num_heads * cache->max_length * head_dim * dtype_size;
        
        cudaMemPool_t mem_pool;
        cudaDeviceGetDefaultMemPool(&mem_pool, key.device().index());
        cudaMallocFromPoolAsync(&cache->key_cache_ptr, cache_size, mem_pool, cache->cache_stream);
        cudaMallocFromPoolAsync(&cache->value_cache_ptr, cache_size, mem_pool, cache->cache_stream);
        
        auto tensor_options = torch::TensorOptions()
            .device(key.device())
            .dtype(key.dtype());
        
        cache->key_tensor = torch::from_blob(
            cache->key_cache_ptr,
            {batch_size, num_heads, cache->max_length, head_dim},
            tensor_options
        );
        
        cache->value_tensor = torch::from_blob(
            cache->value_cache_ptr,
            {batch_size, num_heads, cache->max_length, head_dim},
            tensor_options
        );
        
        cache->key_view = cache->key_tensor;
        cache->value_view = cache->value_tensor;
        
        cache->initialized = true;
    }
    
    // Check if cache needs expansion (rare due to large initial allocation)
    if (cache->current_length + new_seq_len > cache->max_length) {
        int new_max = cache->max_length + std::max(2048, new_seq_len * 8);
        size_t new_cache_size = batch_size * num_heads * new_max * head_dim * dtype_size;
        
        void* new_key_ptr;
        void* new_value_ptr;
        cudaMemPool_t mem_pool;
        cudaDeviceGetDefaultMemPool(&mem_pool, key.device().index());
        cudaMallocFromPoolAsync(&new_key_ptr, new_cache_size, mem_pool, cache->cache_stream);
        cudaMallocFromPoolAsync(&new_value_ptr, new_cache_size, mem_pool, cache->cache_stream);
        
        if (cache->current_length > 0) {
            size_t copy_size = batch_size * num_heads * cache->current_length * head_dim * dtype_size;
            cudaMemcpyAsync(new_key_ptr, cache->key_cache_ptr, copy_size, 
                           cudaMemcpyDeviceToDevice, cache->copy_stream);
            cudaMemcpyAsync(new_value_ptr, cache->value_cache_ptr, copy_size, 
                           cudaMemcpyDeviceToDevice, cache->copy_stream);
        }
        
        cudaFreeAsync(cache->key_cache_ptr, cache->cache_stream);
        cudaFreeAsync(cache->value_cache_ptr, cache->cache_stream);
        
        cache->key_cache_ptr = new_key_ptr;
        cache->value_cache_ptr = new_value_ptr;
        cache->max_length = new_max;
        
        auto tensor_options = torch::TensorOptions()
            .device(key.device())
            .dtype(key.dtype());
        
        cache->key_tensor = torch::from_blob(
            cache->key_cache_ptr,
            {batch_size, num_heads, new_max, head_dim},
            tensor_options
        );
        
        cache->value_tensor = torch::from_blob(
            cache->value_cache_ptr,
            {batch_size, num_heads, new_max, head_dim},
            tensor_options
        );
        
        cache->key_view = cache->key_tensor;
        cache->value_view = cache->value_tensor;
    }
    
    // Use custom kernel for appending data
    size_t total_elements = batch_size * num_heads * new_seq_len * head_dim;
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    
    // Note: Kernel assumes float; modify for float16/bfloat16 if needed
    append_kv_kernel<<<grid, block, 0, cache->copy_stream>>>(
        static_cast<float*>(cache->key_cache_ptr),
        static_cast<float*>(key.data_ptr()),
        batch_size, num_heads, head_dim, new_seq_len, cache->current_length
    );
    
    append_kv_kernel<<<grid, block, 0, cache->copy_stream>>>(
        static_cast<float*>(cache->value_cache_ptr),
        static_cast<float*>(value.data_ptr()),
        batch_size, num_heads, head_dim, new_seq_len, cache->current_length
    );
    
    cache->current_length += new_seq_len;
    
    // Update cached views
    cache->key_view = cache->key_tensor.slice(2, 0, cache->current_length);
    cache->value_view = cache->value_tensor.slice(2, 0, cache->current_length);
    
    return std::make_pair(cache->key_view, cache->value_view);
}

std::pair<torch::Tensor, torch::Tensor> get_cached_kv(int layer_idx) {
    auto it = layer_caches.find(layer_idx);
    if (it != layer_caches.end() && it->second->initialized) {
        auto& cache = it->second;
        return std::make_pair(cache->key_view, cache->value_view);
    }
    
    return std::make_pair(torch::Tensor(), torch::Tensor());
}

void free_cache() {
    layer_caches.clear();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_cache", &init_cache, "Initialize the KV cache with memory pool");
    m.def("update_cache", &update_cache, "Update cache with memory operations", 
          py::arg("key"), py::arg("value"), py::arg("layer_idx"));
    m.def("get_cached_kv", &get_cached_kv, "Get cached KV with operations", 
          py::arg("layer_idx"));
    m.def("free_cache", &free_cache, "Free all cached memory");
}