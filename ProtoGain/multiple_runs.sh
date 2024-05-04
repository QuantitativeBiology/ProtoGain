#!/bin/bash

for run in {1..50}
do
    echo "Running run = $run"
    python3 protogain_test.py --parameters ./breast/parameters.json

done