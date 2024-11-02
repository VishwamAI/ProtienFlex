from setuptools import setup, find_packages

setup(
    name="proteinflex",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.5",
        "scipy==1.9.3",
        "pandas==1.5.3",
        "matplotlib==3.6.2",
        "torch==1.13.1",
        "transformers==4.25.1",
        "tokenizers==0.13.2",
        "biopython==1.80",
        "mdtraj==1.9.7",
        "openmm==7.7.0",
        "prody==2.0.0",
    ],
    extras_require={
        'test': [
            "pytest==7.1.2",
            "pytest-cov==3.0.0",
            "pytest-mock==3.10.0",
            "pytest-asyncio==0.20.3",
            "pytest-timeout==2.1.0",
            "pytest-xdist==3.1.0",
            "pytest-env==0.8.1",
            "pytest-randomly==3.12.0",
            "coverage==6.5.0",
            "flake8==5.0.4",
            "mock==5.0.1"
        ],
    },
    python_requires='>=3.8,<3.9'
)
