#!/bin/bash
eval "$(conda shell.bash hook)"
rm -rf privacy_face_protect/
conda create --prefix privacy_face_protect/ python=3.8 -y
conda activate privacy_face_protect/
pip install -r requirements.txt
