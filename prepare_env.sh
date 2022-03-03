#!/bin/bash

conda create -y -n glnn python=3.6.9
eval "$(conda shell.bash hook)"
conda activate glnn

pip install --no-cache-dir -r requirements.txt
