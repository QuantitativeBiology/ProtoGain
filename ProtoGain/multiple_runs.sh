#!/bin/bash

for run in {1..50}
do
    echo "Running run = $run"
    python3 protogain.py --parameters ./Yasset/parameters.json

done