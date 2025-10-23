from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kv_cache_ext',
    ext_modules=[
        CUDAExtension(
            'kv_cache_ext',
            sources=['kv_cache_ext.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)