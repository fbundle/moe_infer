"""
Build script for moe-infer Cython extension.

Compiles the C wrapper (moe_infer/core_src/moe_infer_c.m, Objective-C) together with
the Cython bridge (moe_infer/core.pyx) into a single shared library.
"""

import os

from helpers.gen_config import generate_default as generate_config
from helpers.gen_shaders import generate as generate_shaders_header

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Regenerate generated headers before each build
_base = os.path.dirname(__file__)
generate_config()
generate_shaders_header(
    os.path.join(_base, "moe_infer", "core_src", "shaders.metal"),
    os.path.join(_base, "moe_infer", "core_src", "shaders.h"),
)

ext = Extension(
    "moe_infer.core",
    sources=[
        "moe_infer/core.pyx",
        "moe_infer/core_src/moe_infer_c.m",
    ],
    include_dirs=["moe_infer/core_src", np.get_include()],
    extra_compile_args=[
        "-O2",
        "-Wall",
        "-fobjc-arc",
        "-DACCELERATE_NEW_LAPACK",
    ],
    extra_link_args=[
        "-lpthread",
        "-lcompression",
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "Accelerate",
    ],
)

setup(
    name="moe-infer",
    version="0.1.0",
    python_requires=">=3.10",
    ext_modules=cythonize(
        [ext],
        language_level="3",
    ),
    packages=["moe_infer", "moe_infer.convert"],
)
