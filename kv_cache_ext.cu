#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <unordered_map>
#include <memory>

// Optimized KV Cache structure with direct memory management
struct LayerKVCache {
    void* key_cache_ptr;    // Raw CUDA memory pointer
    void* value_cache_ptr;  // Raw CUDA memory pointer
    
    // Cache metadata
    int batch_size;
    int num_heads; 
    int head_dim;
    int current_length;
    int max_length;
    int dtype_size;        // Size of data type (float32 = 4, float16 = 2)
    
    // CUDA streams (using default stream for efficiency)
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
        // Don't destroy streams we didn't create
        cache_stream = nullptr;
        copy_stream = nullptr;
        initialized = false;
    }
};

// Global cache storage
static std::unordered_map<int, std::unique_ptr<LayerKVCache>> layer_caches;
static int max_layers = 32;

// Simplified initialization without memory pool overhead
void init_cache() {
    // Clean up existing caches
    layer_caches.clear();
    
    // No memory pool setup - use direct allocations for better performance
}

// Get element size based on PyTorch tensor dtype
int get_dtype_size(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case torch::kFloat32: return 4;
        case torch::kFloat16: return 2;
        case torch::kBFloat16: return 2;
        case torch::kFloat64: return 8;
        default: return 4; // Default to float32
    }
}

// Optimized update function with minimal synchronization
std::pair<torch::Tensor, torch::Tensor> update_cache(
    torch::Tensor key, torch::Tensor value, int layer_idx) {
    
    // Get or create cache for this layer
    if (layer_caches.find(layer_idx) == layer_caches.end()) {
        layer_caches[layer_idx] = std::make_unique<LayerKVCache>();
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
        // Use default CUDA stream instead of creating new ones
        cache->cache_stream = at::cuda::getCurrentCUDAStream();
        cache->copy_stream = cache->cache_stream; // Use same stream
        
        // Initialize cache metadata
        cache->batch_size = batch_size;
        cache->num_heads = num_heads;
        cache->head_dim = head_dim;
        cache->dtype_size = dtype_size;
        cache->current_length = 0;
        
        // Allocate with generous buffer for future tokens
        int buffer_size = std::max(1024, new_seq_len * 8); // Larger initial buffer
        cache->max_length = buffer_size;
        
        size_t cache_size = batch_size * num_heads * buffer_size * head_dim * dtype_size;
        
        // Direct allocation without memory pool overhead
        cudaError_t err1 = cudaMalloc(&cache->key_cache_ptr, cache_size);
        cudaError_t err2 = cudaMalloc(&cache->value_cache_ptr, cache_size);
        
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            throw std::runtime_error("Failed to allocate cache memory");
        }
        
        cache->initialized = true;
    }
    
    // Check if we need to expand the cache
    if (cache->current_length + new_seq_len > cache->max_length) {
        int new_max = cache->max_length * 2; // Simple doubling strategy
        size_t new_cache_size = batch_size * num_heads * new_max * head_dim * dtype_size;
        
        // Allocate new larger buffers
        void* new_key_ptr;
        void* new_value_ptr;
        cudaMalloc(&new_key_ptr, new_cache_size);
        cudaMalloc(&new_value_ptr, new_cache_size);
        
        if (!new_key_ptr || !new_value_ptr) {
            throw std::runtime_error("Failed to expand cache memory");
        }
        
        // Copy existing data to new buffers (synchronous for simplicity)
        if (cache->current_length > 0) {
            size_t copy_size = batch_size * num_heads * cache->current_length * head_dim * dtype_size;
            cudaMemcpy(new_key_ptr, cache->key_cache_ptr, copy_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_value_ptr, cache->value_cache_ptr, copy_size, cudaMemcpyDeviceToDevice);
        }
        
        // Free old memory
        cudaFree(cache->key_cache_ptr);
        cudaFree(cache->value_cache_ptr);
        
        cache->key_cache_ptr = new_key_ptr;
        cache->value_cache_ptr = new_value_ptr;
        cache->max_length = new_max;
    }
    
    // Calculate memory offsets for new data
    size_t element_size = head_dim * dtype_size;
    size_t offset = batch_size * num_heads * cache->current_length * element_size;
    size_t new_data_size = batch_size * num_heads * new_seq_len * element_size;
    
    // Direct memory copy without async overhead for small transfers
    cudaMemcpy(
        static_cast<char*>(cache->key_cache_ptr) + offset,
        key.data_ptr(),
        new_data_size,
        cudaMemcpyDeviceToDevice
    );
    
    cudaMemcpy(
        static_cast<char*>(cache->value_cache_ptr) + offset,
        value.data_ptr(), 
        new_data_size,
        cudaMemcpyDeviceToDevice
    );
    
    cache->current_length += new_seq_len;
    
    // Create PyTorch tensor views from the cached data
    // No synchronization needed since we used synchronous copies
    auto tensor_options = torch::TensorOptions()
        .device(key.device())
        .dtype(key.dtype());
    
    auto cached_key = torch::from_blob(
        cache->key_cache_ptr,
        {batch_size, num_heads, cache->current_length, head_dim},
        tensor_options
    );
    
    auto cached_value = torch::from_blob(
        cache->value_cache_ptr,
        {batch_size, num_heads, cache->current_length, head_dim},
        tensor_options
    );
    
    return std::make_pair(cached_key, cached_value);
}

// Optimized get cached KV without unnecessary synchronization
std::pair<torch::Tensor, torch::Tensor> get_cached_kv(int layer_idx) {
    auto it = layer_caches.find(layer_idx);
    if (it != layer_caches.end() && it->second->initialized) {
        auto& cache = it->second;
        
        // No synchronization needed since we use synchronous operations
        auto tensor_options = torch::TensorOptions()
            .device(torch::kCUDA)
            .dtype(cache->dtype_size == 2 ? torch::kFloat16 : torch::kFloat32);
        
        auto cached_key = torch::from_blob(
            cache->key_cache_ptr,
            {cache->batch_size, cache->num_heads, cache->current_length, cache->head_dim},
            tensor_options
        );
        
        auto cached_value = torch::from_blob(
            cache->value_cache_ptr,
            {cache->batch_size, cache->num_heads, cache->current_length, cache->head_dim},
            tensor_options
        );
        
        return std::make_pair(cached_key, cached_value);
    }
    
    return std::make_pair(torch::Tensor(), torch::Tensor());
}

// Simplified cleanup function
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
