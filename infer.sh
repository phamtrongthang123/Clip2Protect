#!/bin/bash
eval "$(conda shell.bash hook)"
rm ouput_images/*
rm results/*
conda activate encoder4editing
cd encoder4editing
python scripts/inference.py \
--images_dir="/home/ptthang/UARK CLASS/Clip2Protect/input_images" \
--save_dir="/home/ptthang/UARK CLASS/Clip2Protect/ouput_images" \
--latents_only \
--align \
"/home/ptthang/UARK CLASS/Clip2Protect/models/e4e_ffhq_encode.pt"
cd .. 

conda activate clip2protect/
python main.py --data_dir input_images --latent_path ouput_images/latents.pt --protected_face_dir results
