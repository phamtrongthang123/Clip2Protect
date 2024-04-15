#!/bin/bash
eval "$(conda shell.bash hook)"
rm -rf clip2protect/
conda create --prefix clip2protect/ python=3.8 -y
conda activate clip2protect/
pip install -r requirements.txt
