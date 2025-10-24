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
Running benchmark with PyTorch KV-cache...
Running benchmark with Custom KV-cache...

Benchmark Summary
  • WITHOUT cache: 3491.1ms | 901.13 MB
  • PyTorch cache: 1984.1ms | 1375.16 MB
  • Custom cache : 2255.9ms | 1422.49 MB

Speedup Analysis
  • PyTorch cache vs no cache: 1.8x faster
  • Custom cache vs no cache: 1.5x faster
  • Custom cache vs PyTorch cache: 0.9x slower

Memory Usage Comparison
  • PyTorch cache memory reduction: -52.6%
  • Custom cache memory reduction: -57.9%
```