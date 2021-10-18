# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extensions = [
    CUDAExtension(
        'fast_cuda_operator.clib.ngram_repeat.ngram_repeat_block',
        [
            'fast_cuda_operator/clib/ngram_repeat/ngram_repeat_block_cuda.cpp',
            'fast_cuda_operator/clib/ngram_repeat/ngram_repeat_block_cuda_kernel.cu',
        ],
                  ),
]

setup(
    name="fast_cuda_operator",
    version="0.0.1",
    author="Yuxian Meng",
    author_email="yuxian_meng@shannonai.com",
    description="fast C++/CUDA implementations of nlp operations",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP NLG deep learning transformer sequence pytorch tensorflow BERT GPT GPT-2 Microsoft",
    license="MIT",
    url="https://github.com/microsoft/fastseq",
    packages=find_packages(where=".", exclude=["benchmarks", "tests", "__py_cache__"]),
    setup_requires=[
        'torch',
        'cython',
        'setuptools>=18.0',
    ],
    install_requires=[],
    extras_require={},
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension
    },
)
