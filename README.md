# ProtoGain
WORK STILL IN PROGRESS

In this repository you may find a PyTorch implementation of Generative Adversarial Imputation Networks (GAIN) [[1]](#1) for imputing missing iBAQ values in proteomics datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [References](#reference)

## Installation

1. Clone this repository:  `git clone https://github.com/QuantitativeBiology/ProtoGain/`
2. Create a Python environment: `conda create -n proto python=3.10`
3. Activate the previously created environment: `conda activate proto`
4. Install the necessary packages: `pip install -r libraries.txt`


## Usage

The most straightforward and simplest way to run ProtoGain is to run: `python protogain.py -i /path/to/file_to_impute.csv`


However, there are a few arguments which you may want to change. You can do this using a parameters.json file (you may find an example in `ProtoGain/breast/parameters.json`) or you can choose them directly in the command line.

Run with a parameters.json file: `python protogain.py --parameters /path/to/parameters.json`
Run with command line arguments: `python protogain.py -i /path/to/file_to_impute.csv -o imputed_name --ofolder ./results/ --it 2001`

#### Arguments:

`-i`: Path to file to impute<br> 
`-o`: Name of imputed file<br> 
`--ofolder`: Path to the output folder<br> 
`--it`: Number of iterations to train the model<br> 
`--outall`: Set this argument to `1` if you want to output every metric<br> 
`--override`: Set this argumento to `1` if you want to delete the previously created files when writing the new output<br> 

WORK IN PROGRESS



## References
<a id="1">[1]</a> 
J. Yoon, J. Jordon & M. van der Schaar (2018). GAIN: Missing Data Imputation using Generative Adversarial Nets
