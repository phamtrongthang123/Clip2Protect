from gender_detector import get_gender
import os 
import argparse
prompt_bank = {'Woman': 'red lipstick with pink eyeshadows', 'Man': 'no makeup'}

# get img path from args
parser = argparse.ArgumentParser(description="CLIP2Protect")
parser.add_argument('--image_path', type=str, default='047073.jpg', help='The input image')
args = parser.parse_args()
image_path = args.image_path
# get gender
gender = get_gender(image_path)
current_prompt = prompt_bank[gender]

# copy img to a folder
os.system('rm -rf auto_input_images')
os.system('mkdir -p auto_input_images')
os.system(f'cp {image_path} auto_input_images')

# run get latent 
os.system('bash infer_part1.sh')


# choose prompt and run main.py (go into this script) 
os.system(f'python main.py --data_dir auto_input_images --latent_path auto_ouput_images/latents.pt --protected_face_dir results --makeup_prompt "{current_prompt}"')

