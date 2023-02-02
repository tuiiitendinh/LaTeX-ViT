from pix2tex.dataset.dataset import Im2LatexDataset
import os
import argparse
import logging
import yaml

import torch
from munch import Munch
from tqdm.auto import tqdm
import wandb
import torch.nn as nn
from pix2tex.eval import evaluate
from pix2tex.models import get_model
# from pix2tex.utils import *
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check

def validation_testing(args, dataloader, e = None):
    model = get_model(args)
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.name, "test.pth")), map_location=args.device)
    
    if e is not None:
        num_batches = int(args.valbatches*e/args.epochs)
        name = 'val'
    else:
        num_batches = args.testbatchsize
        name = 'test'

    with torch.no_grad():
        bleu_score, edit_distance, token_accuracy = evaluate(model, dataloader, args, num_batches=num_batches, name=name)

    if args.wandb:
        wandb.log({f'{name}/bleu': bleu_score, f'{name}/edit_distance': edit_distance, f'{name}/token_accuracy': token_accuracy})

def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args, test=False)
    valdataloader = Im2LatexDataset().load(args.valdata)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)

    # add testdata in config file
    testloader = Im2LatexDataset().load(args.testdata)
    testloader.update(**args, test=False)

    device = args.device
    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)
    # val_max_bleu, val_max_token_acc = 0, 0
    # test_max_bleu, test_max_token_acc = 0, 0
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device))

    def save_models():
        # if test:
        #     filename = os.path.join(out_path, '%s_e%02d_step%02d_test.pth' % (args.name, e+1, step))
        # else:
        #     filename = os.path.join(out_path, '%s_e%02d_step%02d.pth' % (args.name, e+1, step))
        filename = os.path.join(out_path, "test.pth")
        # print("Name: ", filename)
        # print("Model: ", model.state_dict())
        torch.save(model.state_dict(), filename)
        # torch.save(model.state_dict(), "test_1.pth")
        
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))
        print("Saved model at: ", filename)
        
    if args.optimizer == 'Adam':
        opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas)
    elif args.optimizer == 'AdamW':
        opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    
    if args.scheduler == 'StepLR':
        scheduler = get_scheduler(args.scheduler)(opt, step_size=args.lr_step, gamma=args.gamma)
    elif args.scheduler == 'OneCycleLR':
        scheduler = get_scheduler(args.scheduler)(opt, max_lr=args.max_lr, pct_start=args.pct_start, total_steps=round(int(len(dataloader)*args.epochs)))

    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize

    test_counter = 0

    try:
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(dataloader))

            #resets the memory allocation tracker at the beginning of each epoch
            torch.cuda.reset_max_memory_allocated()

            for i, (seq, im) in enumerate(dset):    
                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0
                    for j in range(0, len(im), microbatch):
                        tgt_seq, tgt_mask = seq['input_ids'][j:j+microbatch].to(device), seq['attention_mask'][j:j+microbatch].bool().to(device)
                        loss = model.data_parallel(im[j:j+microbatch].to(device), device_ids=args.gpu_devices, tgt_seq=tgt_seq, mask=tgt_mask)*microbatch/args.batchsize
                        loss.backward()  # data parallism loss is a vector
                        total_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    opt.step()
                    scheduler.step()
                    dset.set_description('Loss: %.4f' % total_loss)
                    if args.wandb:
                        wandb.log({'train/loss': total_loss})
                        wandb.log({'train/lr': scheduler.get_last_lr()[0]})
                    
                    #releases unoccupied memory at the end of each iteration
                    torch.cuda.empty_cache()  

                #save model after every epoch            
                # if (e+1) % args.save_freq == 0:

                if (i+1+len(dataloader)*e) % args.sample_freq == 0:
                    test_counter += 1
                    validation_testing(args, valdataloader, e)

                    
                        
                #test model on testing set each 5 times after validation test
                if test_counter == 5:
                    validation_testing(args, testloader)
                    test_counter = 0
                    # with torch.no_grad():
                    #     model.eval()
                    #     bleu_score_test, edit_distance_test, token_accuracy_test = evaluate(model, testloader, args, num_batches=args.testbatchsize, name='test')
                    #     if bleu_score_test > test_max_bleu and token_accuracy_test > test_max_token_acc:
                    #         test_max_bleu, test_max_token_acc = bleu_score_test, token_accuracy_test
                    #                                     save_models(e, step=i, test = True)  
                    #     test_counter = 0
                    # model.train()

            #save model after every epochs
            save_models()
            if args.wandb:
                wandb.log({'train/epoch': e+1})
    except KeyboardInterrupt:
        if e >= 2:
            save_models()
        raise KeyboardInterrupt
    save_models()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--resume', help='path to checkpoint folder', action='store_true')
    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/debug.yaml')
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)
    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), resume='allow', name=args.name, id=args.id)
        args = Munch(wandb.config)
    train(args)