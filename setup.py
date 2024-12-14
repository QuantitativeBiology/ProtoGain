from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    description = f.read()


setup(
    name="ProtoGen",
    version="0.2.0",
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
        "psutil",
    ],  
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Minimum Python version
    license="BSD-3-Clause",  
   
   # entry_points={
    #    "console_scripts": [
     #       "protogain=ProtoGain.protogain:main",  # CLI command to run the main function
      #  ],
    #},
)