from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='1.0.0',
    author='anonymous',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='anonymous@anonymous.edu',
    description='Extended legged_gym environment with reward machine',
    install_requires=['isaacgym',
                      'rm_ppo',
                      'matplotlib']
)