#import ProtoGain

# Call some method that produces an output
#args = ProtoGain.init_arg()
#print(f"Initialized arguments: {args}")

from ProtoGain import create_csv  
from ProtoGain import sample_idx

data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
name = "test_file"
header = ["Column1", "Column2", "Column3"]

create_csv(data, name, header)

print(f"CSV file '{name}.csv' created successfully!")
print(sample_idx(10, 5)) 


from ProtoGain import utils

import ProtoGain.utils as utils
print(utils.__file__)

import ProtoGain.utils as utils
print(dir(utils))  # To list all available attributes in `utils`
import torch
from ProtoGain import Network
from ProtoGain import Params
from ProtoGain import Metrics
from ProtoGain import Data
import pandas as pd
import numpy as np

def test_network():
    # Load the dataset
    dataset_path = "./ProtoGain/breast/breastMissing_20.csv"  # Input dataset with missing values
    ref_path = "./ProtoGain/breast/breast.csv"  # Reference complete dataset
    
    # Load dataset and reference
    dataset_df = pd.read_csv(dataset_path)
    dataset = dataset_df.values  # Convert to numpy array
    ref = pd.read_csv(ref_path).values  # Reference dataset
    
    # Extract headers (missing_header)
    missing_header = dataset_df.columns.tolist()

    # Define mock parameters for testing
    params = Params(
        input=dataset_path,
        output="imputed.csv",
        ref=ref_path,
        output_folder=".",
        num_iterations=2001,
        batch_size=128,
        alpha=10,
        miss_rate=0.1,
        hint_rate=0.9,
        lr_D=0.001,
        lr_G=0.001,
        override=1,
        output_all=1,
    )
    
    # Dummy network structures based on input dimensions
    input_dim = dataset.shape[1]  # Number of features
    h_dim = input_dim  # Hidden layer size
    net_G = torch.nn.Sequential(
        torch.nn.Linear(input_dim * 2, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, input_dim),
        torch.nn.Sigmoid()
    )
    net_D = torch.nn.Sequential(
        torch.nn.Linear(input_dim * 2, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, input_dim),
        torch.nn.Sigmoid()
    )
    
    # Initialize metrics
    metrics = Metrics(params)

    # Initialize the Network
    network = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)
    print("Network initialized successfully!")

    # Initialize Data
    data = Data(
        dataset=dataset,
        miss_rate=0.2,
        hint_rate=0.9,
        ref=ref  # Provide reference if available
    )
    
    # Perform training (imputation)
    print("Running imputation...")
    try:
        network.train(data=data, missing_header=missing_header)  # Pass missing_header
        print("Imputation completed successfully!")
    except Exception as e:
        print(f"Error during imputation: {e}")

if __name__ == "__main__":
    test_network()





