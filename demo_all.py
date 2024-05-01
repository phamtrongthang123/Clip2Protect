# python demo.py --image_path 047073.jpg
from gender_detector import get_gender
import os 
import argparse
from pathlib import Path 
from tqdm  import tqdm 
input_dir = '/home/ptthang/UARK CLASS/privacy_face_protect/Images from dataset/Images from dataset'
saved_dir = '/home/ptthang/UARK CLASS/privacy_face_protect/Images from dataset/Images from dataset_new'
os.makedirs(saved_dir, exist_ok=True)
all_images = [str(x) for x in Path(input_dir).rglob('*.jpg')]
prompt_bank = {'Man': 'red lipstick with pink eyeshadows', 'Woman': 'no makeup'}
for image_path in tqdm(all_images):
    # get gender
    gender = get_gender(image_path)
    current_prompt = prompt_bank[gender]

    # copy img to a folder
    os.system('rm -rf auto_input_images')
    os.system('mkdir -p auto_input_images')
    os.system(f'cp "{image_path}" auto_input_images')

    # run get latent 
    os.system('bash infer_part1.sh')

    # choose prompt and run main.py (go into this script) 
    os.system(f'python main.py --data_dir auto_input_images --latent_path auto_ouput_images/latents.pt --protected_face_dir "{saved_dir}" --makeup_prompt "{current_prompt}"')

