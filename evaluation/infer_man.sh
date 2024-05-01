#!/bin/bash
eval "$(conda shell.bash hook)"
cd "/home/ptthang/UARK CLASS/privacy_face_protect"
rm ouput_images/*
rm CelebA_HQ_face_gender_dataset_test_1000_new/man/*

conda activate encoder4editing
cd encoder4editing
python scripts/inference.py \
--images_dir="/home/ptthang/UARK CLASS/privacy_face_protect/CelebA_HQ_face_gender_dataset_test_1000/man" \
--save_dir="/home/ptthang/UARK CLASS/privacy_face_protect/ouput_images" \
--latents_only \
--align \
"/home/ptthang/UARK CLASS/privacy_face_protect/models/e4e_ffhq_encode.pt"
cd .. 

mkdir -p CelebA_HQ_face_gender_dataset_test_1000_new/man

conda activate privacy_face_protect/
python main.py --data_dir "/home/ptthang/UARK CLASS/privacy_face_protect/CelebA_HQ_face_gender_dataset_test_1000/man" --latent_path ouput_images/latents.pt --protected_face_dir CelebA_HQ_face_gender_dataset_test_1000_new/man
