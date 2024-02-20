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
from torch.optim.lr_scheduler import OneCycleLR
# from pix2tex.utils import *
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler
from pix2tex.utils.seq_loss import SequenceLoss

def train(args):
    tokenizer_path = "pix2tex/model/dataset/tokenizer.json"

    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args, test=False)
    
    valdataloader = Im2LatexDataset().load(args.valdata)
    valargs = args.copy()
    valargs.update(batchsize=args.valbatches, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)

    # add testdata in config file
    testargs = args.copy()
    testloader = Im2LatexDataset().load(args.testdata)
    testargs.update(batchsize=args.valbatches, keep_smaller_batches=True)
    testloader.update(**args, test=False)

    # print(f"--- Validloader Length: {len(valdataloader)}")
    # print(f"--- Testloader Length: {len(testloader)}")

    
    device = args.device
    model = get_model(args)
    # if torch.cuda.is_available() and not args.no_cuda:
    #     gpu_memory_check(model, args)
    val_max_bleu, val_max_token_acc = 0, 0
    test_max_bleu, test_max_token_acc = 0, 0
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device))

    def save_models(e, bleu, test = False):
        if test:
            filename = os.path.join(out_path, '%s_e%02d_bleu%02f_test.pth' % (args.name, e+1, bleu))
        else:
            filename = os.path.join(out_path, '%s_e%02d_bleu%02f.pth' % (args.name, e+1, bleu))
        
        # print("Name: ", filename)
        # print("Model: ", model.state_dict())
        torch.save(model.state_dict(), filename)
        # torch.save(model.state_dict(), "test_1.pth")
        
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))
        print(f"Saved best model with BLEU {bleu:.2f} at: {filename}")
        
    if args.optimizer == 'Adam':
        opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas)
    elif args.optimizer == 'AdamW':
        opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    
    if args.scheduler == 'StepLR':
        scheduler = get_scheduler(args.scheduler)(opt, step_size=args.lr_step, gamma=args.gamma)
    elif args.scheduler == 'OneCycleLR':
        scheduler = get_scheduler(args.scheduler)(opt, max_lr=args.lr, 
                                                  pct_start=args.pct_start, 
                                                  total_steps=round(int(len(dataloader)*args.epochs)), 
                                                  anneal_strategy=args.anneal_strategy,
                                                  div_factor=args.div_factor,
                                                  final_div_factor=args.final_div)

    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize

    test_counter = 0
    
    seq_loss = SequenceLoss(dataloader.tokenizer)

    try:
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(dataloader))
            # print(f"--- Dataloader Length: {len(dset)}")

            #resets the memory allocation tracker at the beginning of each epoch
            torch.cuda.reset_max_memory_allocated()

            for i, (seq, im) in enumerate(dset):    
                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0
                    for j in range(0, len(im), microbatch):
                        tgt_seq, tgt_mask = seq['input_ids'][j:j+microbatch].to(device), seq['attention_mask'][j:j+microbatch].bool().to(device)
                        
                        # Gradient of Log Probability with respect to the model parameters \theta
                        loss_grad = model.data_parallel(im[j:j+microbatch].to(device), device_ids=args.gpu_devices, tgt_seq=tgt_seq, mask=tgt_mask) * microbatch/args.batchsize

                        #tgt_seq is the label of data, only create the prediction for optimizing
                        y_hat = model.generate(im.to(device), temperature=args.get('temperature', .2))

                        loss_grad = seq_loss(tgt_seq, y_hat, loss_grad)
                        # print(f"Loss: {loss_grad}\n")
                        loss_grad.backward()  # data parallism loss is a vector
                        total_loss += loss_grad.item()
                        # print(f"Total Loss: {total_loss}\n")
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        
                    opt.step()
                    scheduler.step()
                    dset.set_description('Loss: %.4f' % total_loss)
                    if args.wandb:
                        wandb.log({'train/loss': total_loss})
                        wandb.log({'train/lr': scheduler.get_last_lr()[0]})  

                if (i+1+len(dataloader)*e) % args.sample_freq == 0:
                    #validation testing
                    test_counter += 1
                    with torch.no_grad():
                        model.eval()
                        bleu_score_val, _, token_accuracy_val = evaluate(model, valdataloader, args, num_batches=args.valbatches, name='val')
                        if bleu_score_val > val_max_bleu and token_accuracy_val > val_max_token_acc:
                            val_max_bleu, val_max_token_acc = bleu_score_val, token_accuracy_val
                            save_models(e, val_max_bleu, test = False)
                    model.train()
                        
                #test model on testing set each 5 times after validation test
                if test_counter == 4:
                    with torch.no_grad():
                        model.eval()
                        bleu_score_test, edit_distance_test, token_accuracy_test = evaluate(model, testloader, args, num_batches=args.testbatchsize, name='test')
                        if bleu_score_test > test_max_bleu and token_accuracy_test > test_max_token_acc:
                            test_max_bleu, test_max_token_acc = bleu_score_test, token_accuracy_test
                            if args.wandb:
                                wandb.log({'test_periodically/bleu': bleu_score_test, 'test_periodically/edit_distance': edit_distance_test, 'test_periodically/token_accuracy': token_accuracy_test})
                            save_models(e, test_max_bleu, test = True)  
                        test_counter = 0
                    model.train()

            #save model after every epoch            
            # if (e+1) % args.save_freq == 0:
            #     save_models(e, step=len(dataloader), test = False)
            if args.wandb:
                wandb.log({'train/epoch': e+1})
    except KeyboardInterrupt:
        # if e >= 2:
            # save_models(e, val_max_bleu, test = False) 
        raise KeyboardInterrupt
    # save_models(e, test = False)
    


if __name__ == '__main__':
    import os
    
    os.environ["WANDB_API_KEY"] = '2ddfeaa30e5503d98ac09682e44871f2b80b0f6b'
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

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