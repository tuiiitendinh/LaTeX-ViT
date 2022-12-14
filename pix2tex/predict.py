import yaml
import base64
import cv2
import numpy as np
import torch

from transformers import PreTrainedTokenizerFast
from pix2tex.dataset.transforms import test_transform
from pix2tex.models import get_model
from pix2tex.utils import post_process, token2str
from munch import Munch


config_pth = '/content/drive/MyDrive/LaTeX-OCR/checkpoints/im2latex_weai_ver4/config.yaml'
tokenizer_pth = '/content/drive/MyDrive/LaTeX-OCR/pix2tex/model/dataset/weai_tokenizer.json'
model_pth = '/content/drive/MyDrive/LaTeX-OCR/checkpoints/im2latex_weai_ver4/im2latex_weai_ver4_e22_step1389.pth'

with open(config_pth) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
args = Munch(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
args.wandb = False

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_pth)

#get model
model = get_model(args)
model.load_state_dict(torch.load(model_pth, map_location=device))
model.eval()

def pad_w(img, pad_w):
    left_pad =32
    right_pad = pad_w + 16
    return np.copy(np.pad(img, ((0, 0), (left_pad, right_pad), (0,0)), mode='constant', constant_values=255))

def pad_h(img, pad_h):
    top_pad = 32
    bottom_pad = pad_h + 16
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (0, 0), (0,0)), mode='constant', constant_values=255))

#define a predict funciton:
#input: base64 encoded image

def predict(img):
    # img = base64.b64decode(img)
    # im = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_UNCHANGED)
    im = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    height, width = im.shape[0], im.shape[1]
    
    # print(height, width)

    pad_width, pad_height = 0, 0
    if height % 16 != 0:
        pad_height = 16 - (height % 16)
        im = pad_h(im, pad_height)

    if width % 16 != 0:
        pad_width = 16 - (width % 16)
        im = pad_w(im, pad_width)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # print(im.shape)

    img_pred = test_transform(image=im)['image'][:1].unsqueeze(0)
    # print(img_pred.shape)

    dec = model.generate(img_pred.to(args.device), temperature=args.get('temperature', .2)) #tensor
    print(dec)
    pred = post_process(token2str(dec, tokenizer)[0]) #str
    print(pred)
    return pred