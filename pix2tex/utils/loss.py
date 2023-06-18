import torch
from torchtext.data import metrics
from torch import nn
from pix2tex.eval import detokenize
from pix2tex.utils import alternatives

#init a SequenceLoss class, inherit from nn.Module, and take the tokenizers as arguments
class SequenceLoss(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    #define the forward pass, which takes the y and y_hat tensors as arguments
    def forward(self, y, y_hat):
        y = detokenize(y, self.tokenizer)
        # print("utils/loss ... tgt_seq: ", y)            
        y_hat = detokenize(y_hat, self.tokenizer)
        # print("utils/loss ... pred_seq: ", y_hat)
        
        #calculate the reward score
        reward = metrics.bleu_score(y_hat, [alternatives(x) for x in y])
        print("Reward BLEU: ", reward)
        return reward