from setuptools import setup, find_packages

setup(
    name="proteinflex",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "torch>=2.0.0,<3.0.0",
        "transformers>=4.36.0",
        "biopython>=1.79,<2.0.0",
    ],
    extras_require={
        'test': [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0"
        ],
    }
)
