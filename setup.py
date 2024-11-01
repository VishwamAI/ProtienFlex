# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup, find_packages

setup(
    name="proteinflex",
    version="1.0.0",
    description="AI-driven protein development and analysis toolkit",
    author="VishwamAI",
    author_email="contact@vishwamai.com",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.2.0',
        'biopython>=1.79',
        'mdtraj>=1.9.7',
        'prody>=2.4.0',
        'requests>=2.31.0',
        'tqdm>=4.66.0',
        'matplotlib>=3.8.0',
        'seaborn>=0.13.0',
        'pytest>=8.0.0',
        'pytest-cov>=4.1.0',
        'jax>=0.4.20',
        'jaxlib>=0.4.20',
        'dm-haiku>=0.0.10',
        'tensorflow>=2.15.0',
        'openmm>=8.1.1',
    ],
    extras_require={
        'dev': [
            'black',
            'flake8',
            'isort',
            'mypy',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
            'myst-parser',
        ],
    },
    python_requires='>=3.10,<3.11',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'proteinflex=proteinflex.cli:main',
        ],
    },
)
