import os
import sys
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


class PostInstallation(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Note: buggy for kornia==0.5.3 and it will be fixed in the next version.
        # Set kornia to 0.5.2 temporarily
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'kornia==0.5.2', '--no-dependencies'])


if __name__ == '__main__':
    # version = '0.3.0+%s' % get_git_commit_number()
    # write_version_to_file(version, 'version.py')

    setup(
        name='custom_test',
        # version=version,
        # description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        # install_requires=[
        #     'numpy',
        #     'torch>=1.1',
        #     'numba',
        #     'tensorboardX',
        #     'easydict',
        #     'pyyaml'
        # ],
        # author='Shaoshuai Shi',
        # author_email='shaoshuaics@gmail.com',
        # license='Apache License 2.0',
        packages=find_packages(exclude=['PyTorchCustomOperator']),
        cmdclass={
            'build_ext': BuildExtension,
            # 'install': PostInstallation,
            # Post installation cannot be done. ref: https://github.com/pypa/setuptools/issues/1936.
            # 'develop': PostInstallation,
        },
        ext_modules=[
            make_cuda_ext(
                name='fps',
                module='.',
                sources=[
                    'sampling.cpp',
                    'sampling_gpu.cu',
                ],
            ),
        ],
    )
