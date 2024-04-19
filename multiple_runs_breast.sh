#!/bin/bash

# List of parameters
runs=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45" "46" "47" "48" "49")
train_samples=("455" "409" "364" "318" "273" "227" "204" "182" "159" "136")
train_samples=("136" "159" "182" "204" "227" "273" "318" "364" "409" "455")

# Loop over parameters
for samples in "${train_samples[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running with samples = $samples and run = $run"
        python3 my_gain_downsampling.py --run $run --samples $samples
    done
done