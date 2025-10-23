# KV Cache extension for pytorch

## Installation

```bash
conda create -n kv_cache_ext python=3.10 -y
conda activate kv_cache_ext
pip install -r requirements.txt
# Please ensure pytorch-cuda is installed for the correct CUDA version
```

```bash # To build the C++/CUDA extension
python setup.py build_ext --inplace
```

## Usage

```bash
python3 benchmark_kv_cache.py
```

## Benchmark Results

```terminal
# Example benchmark results
Running benchmark without KV-cache...
🐌 PyTorch WITHOUT cache: 1858.3ms, 785.10 MB
Running benchmark with PyTorch KV-cache...
🔥 PyTorch WITH cache: 1277.3ms, 965.68 MB
Running benchmark with custom CUDA KV-cache...
🧠 Custom CUDA KV-cache: 1470.8ms, 989.38 MB

⚖️ Benchmark Summary
  • WITHOUT cache: 1858.3ms | 785.10 MB
  • PyTorch cache: 1277.3ms | 965.68 MB
  • Custom cache : 1470.8ms | 989.38 MB

🚀 Speedup Analysis
  • PyTorch cache vs no cache: 1.5x faster
  • Custom cache vs no cache: 1.3x faster
  • Custom cache vs PyTorch cache: 0.9x slower

💾 Memory Usage Comparison
  • PyTorch cache memory reduction: -23.0%
  • Custom cache memory reduction: -26.0%
```