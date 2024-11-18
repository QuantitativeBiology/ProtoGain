from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    description = f.read()


setup(
    name="ProtoGAIN",  
    version="0.1.0",  
    author="Diogo Ferreira, Emanuel Gonçalves, Jorge Ribeiro, Leandro Sobral, Rita Gama",  
    author_email="quantitative-biology@googlegroups.com",  
    description="A Python package for synthetic proteomics data augmentation using ProtoGAIN",
    long_description=description,  # Use the content of README.md
    long_description_content_type="text/markdown",  # Markdown format for PyPI rendering
    url="https://github.com/QuantitativeBiology/ProtoGain",  # Replace with your GitHub repository URL
    packages=find_packages(),  # Automatically find all packages (like 'protogain')
    install_requires=[
        "torch",
        "torchinfo",
        "numpy",
        "tqdm",
        "pandas",
        "scikit-learn",
        "optuna",
        "argparse",
        "psutil",
    ],  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
    license="BSD-3-Clause",  
    entry_points={
        "console_scripts": [
            "protogain=protogain.protogain:main",  # CLI command to run the main function
        ],
    },
)
