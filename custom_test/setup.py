import os
import sys
import subprocess
from setuptools.command.install import install
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, load
from setuptools import setup

if __name__ == '__main__':
    setup(name='fps',include_dirs = ["./"],
        ext_modules=[CUDAExtension('fps',[
            'sampling.cpp',
            'sampling_gpu.cu'
            ],),],
    cmdclass={'build_ext': BuildExtension})
