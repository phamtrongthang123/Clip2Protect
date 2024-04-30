# From https://github.com/CGCL-codes/AMT-GAN/blob/main/test.py


import os
import time
import requests
from json import JSONDecoder
from PIL import Image
from tqdm import tqdm
from datasets.assets.models import irse, ir152, facenet
from backbone import Inference
from backbone import PostProcess
from setup import setup_config, setup_argparser
import torch.nn.functional as F
from utils import *



def attack_local_models(attack=True):
    parser = setup_argparser()
    parser.add_argument("--model_path", default="assets/models", help="model path")
    parser.add_argument("--model_names", type=list, default=['mobile_face'], help="model for testing")
    parser.add_argument("--save_path", default="assets/datasets/save", help="path to generated images")
    parser.add_argument("--clean_path", default="assets/datasets/CelebA-HQ", help="path to clean images")
    parser.add_argument("--target_path", default="assets/datasets/test/047073.jpg", help="path to target images")
    parser.add_argument('--device', type=str, default='cuda', help='cuda device')
    args = parser.parse_args()
    test_models = {}
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}
    for model_name in args.model_names:
        if model_name == 'ir152':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load('./assets/models/ir152.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'irse50':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load('./assets/models/irse50.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'facenet':
            test_models[model_name] = []
            test_models[model_name].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=args.device)
            fr_model.load_state_dict(torch.load('./assets/models/facenet.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'facenet':
            test_models[model_name] = []
            test_models[model_name].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=args.device)
            fr_model.load_state_dict(torch.load('./assets/models/facenet.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'mobile_face':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load('./assets/models/mobile_face.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
    for test_model in test_models.keys():
        size = test_models[test_model][0]
        model = test_models[test_model][1]
        target = read_img(args.target_path, 0.5, 0.5, args.device) # ??? 
        target_embbeding = model((F.interpolate(target, size=size, mode='bilinear')))
        FAR01 = 0
        FAR001 = 0
        FAR0001 = 0
        total = 0

        for img in tqdm(os.listdir(args.clean_path), desc=test_model + ' clean'):
            adv_example = read_img(os.path.join(args.clean_path, img), 0.5, 0.5, args.device)
            ae_embbeding = model((F.interpolate(adv_example, size=size, mode='bilinear')))
            cos_simi = torch.cosine_similarity(ae_embbeding, target_embbeding)
            if cos_simi.item() > th_dict[test_model][0]:
                FAR01 += 1
            if cos_simi.item() > th_dict[test_model][1]:
                FAR001 += 1
            if cos_simi.item() > th_dict[test_model][2]:
                FAR0001 += 1
                total += 1
        print(test_model, "ASR in FAR@0.1: {:.4f}, ASR in FAR@0.01: {:.4f}, ASR in FAR@0.001: {:.4f}".
              format(FAR01/total, FAR001/total, FAR0001/total))




if __name__ == '__main__':
    generate()
    attack_local_models(attack=False)
    attack_local_models()

