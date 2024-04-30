#!/bin/bash
eval "$(conda shell.bash hook)"
rm auto_ouput_images/*
mkdir -p auto_ouput_images
rm results/*
conda activate encoder4editing
cd encoder4editing
python scripts/inference.py \
--images_dir="/home/ptthang/UARK CLASS/privacy_face_protect/auto_input_images" \
--save_dir="/home/ptthang/UARK CLASS/privacy_face_protect/auto_ouput_images" \
--latents_only \
--align \
"/home/ptthang/UARK CLASS/privacy_face_protect/models/e4e_ffhq_encode.pt"
cd .. 