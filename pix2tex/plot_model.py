from pix2tex.dataset.dataset import Im2LatexDataset
import argparse
import logging
import yaml

import numpy as np
import torch
from munch import Munch

from pix2tex.models import get_model, Model
from pix2tex.utils import *
from torchviz import make_dot
from torchsummary import summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('-c', '--checkpoint', default=None, type=str, help='path to model checkpoint')

    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/config.yaml')
            
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))
    args.wandb = False
    seed_everything(args.seed if 'seed' in args else 42)
    model = get_model(args)
    model.load_state_dict(torch.load(parsed_args.checkpoint, args.device))
    print(model)
    # x = summary(model,
    #         (
    #           args.channels,
    #           args.max_height,
    #           args.max_width,
    #           args.min_height,
    #           args.min_width,
    #           1182,
    #         ),
    #         10, 
    #       )
    # print(x)

    # make_dot(model, params=dict(list(model.named_parameters()))).render("im2tex_architecture", format="png")