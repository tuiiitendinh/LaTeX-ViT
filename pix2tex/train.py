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
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check
from pix2tex.utils.loss import SequenceLoss

def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args, test=False)
    valdataloader = Im2LatexDataset().load(args.valdata)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)

    testloader = Im2LatexDataset().load(args.testdata)
    testloader.update(**args, test=True)

    device = args.device
    model = get_model(dataloader, args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)
    val_max_bleu, val_max_token_acc = 0, 0
    test_max_bleu, test_max_token_acc = 0, 0
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device))

    def save_models(e, step=0, test = False, last_epoch = False):
        if test:
            filename = os.path.join(out_path, '%s_e%02d_step%02d_test.pth' % (args.name, e+1, step))
        else:
            filename = os.path.join(out_path, '%s_e%02d_step%02d.pth' % (args.name, e+1, step))
            
        if last_epoch:
            filename = os.path.join(out_path, 'final_model.pth')

        torch.save(model.state_dict(), filename)

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

    seq_loss = SequenceLoss(dataloader.tokenizer)

    try:
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(dataloader))

            torch.cuda.reset_max_memory_allocated(device=device)
            
            for i, (seq, im) in enumerate(dset):
                torch.cuda.empty_cache()

                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0

                    for j in range(0, len(im), microbatch):
                        tgt_seq, tgt_mask = seq['input_ids'][j:j+microbatch].to(device), seq['attention_mask'][j:j+microbatch].bool().to(device)
                        parallel_loss = model.data_parallel(im[j:j+microbatch].to(device), device_ids=args.gpu_devices, tgt_seq=tgt_seq, mask=tgt_mask)*microbatch/args.batchsize
                        
                        # y: label, y_hat: prediction, y_multi_distr: multinimial distribution of predictions in a batch
                        y, y_hat, y_multi_distr = tgt_seq, model.generate(im.to(device), temperature=args.get('temperature', .2))
                        loss = seq_loss(y, y_hat, parallel_loss)
                        
                        
                        loss.backward()  
                        total_loss += loss.item()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    #                 opt.step()

    #                 scheduler.step()

    #                 dset.set_description('Loss: %.4f' % total_loss)

    #                 if args.wandb:
    #                     wandb.log({'train/loss': total_loss})
    #                     wandb.log({'train/lr': scheduler.get_last_lr()[0]})

    #             torch.cuda.empty_cache()

    #             if (i+1+len(dataloader)*e) % args.sample_freq == 0:
    #                 # Switch to evaluation mode
    #                 model.eval()
    #                 # Validate model
    #                 with torch.no_grad():
    #                     bleu_score_val, edit_distance_val, token_accuracy_val = evaluate(model, valdataloader, args, num_batches=round(int(args.valbatches*e/args.epochs)), name='val')

    #                     # If current validation scores are the best so far, save the model
    #                     if bleu_score_val > val_max_bleu and token_accuracy_val > val_max_token_acc:
    #                         val_max_bleu, val_max_token_acc = bleu_score_val, token_accuracy_val
    #                         save_models(e, step=i, test = False, last_epoch = False)
    #                 # Switch back to training mode
    #                 model.train()

    #             # Test the model on the test set periodically after a certain number of validation tests
    #             #torch.cuda.empty_cache()
    #             if test_counter == 4 :
    #                 model.eval()
    #                 with torch.no_grad():
    #                     bleu_score_test, edit_distance_test, token_accuracy_test = evaluate(model, testloader, args, num_batches=args.testbatchsize, name='test')
    #                     if bleu_score_test > test_max_bleu and token_accuracy_test > test_max_token_acc:
    #                         test_max_bleu, test_max_token_acc = bleu_score_test, token_accuracy_test
    #                         # if args.wandb:
    #                         #     wandb.log({'test_periodically/bleu': bleu_score_test, 'test_periodically/edit_distance': edit_distance_test, 'test_periodically/token_accuracy': token_accuracy_test})
    #                         save_models(e, step=i, test = True, last_epoch = False)
    #                     test_counter = 0
    #                 model.train()

            
    #         # #test model after every epoch
    #         # # test_counter += 1
    #         # # if test_counter == 4:
    #         # #     model.eval()
    #         # #     with torch.no_grad():
    #         # #         bleu_score_test, edit_distance_test, token_accuracy_test = evaluate(model, testloader, args, num_batches=args.testbatchsize, name='test')
    #         # #         if bleu_score_test > test_max_bleu and token_accuracy_test > test_max_token_acc:
    #         # #             test_max_bleu, test_max_token_acc = bleu_score_test, token_accuracy_test
    #         # #             # if args.wandb:
    #         # #             #     wandb.log({'test_periodically/bleu': bleu_score_test, 'test_periodically/edit_distance': edit_distance_test, 'test_periodically/token_accuracy': token_accuracy_test})
    #         # #             save_models(e, step=i, test = True, last_epoch = False)
    #         # #         test_counter = 0
    #         # #     model.train()

    #         # # Save the model at the end of each 'save_freq' epochs
    #         # if (e+1) % args.save_freq == 0:
    #         #     save_models(e, step=len(dataloader), test = False, last_epoch = False)

    #         #save model after every epoch
    #         if (e+1) % args.save_freq == 0:
    #             save_models(e, step=len(dataloader), test = False, last_epoch = False)
    #         if args.wandb:
    #             wandb.log({'train/epoch': e+1})

    except KeyboardInterrupt:
        pass
    #     if e >= 2:
    #         save_models(e, step=i, last_epoch = False)
    #     raise KeyboardInterrupt
    # save_models(e, step=len(dataloader), test = False, last_epoch = True)


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