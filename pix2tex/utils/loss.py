import torch
from torchtext.data import metrics
from torch import nn
from pix2tex.eval import detokenize

#init a SequenceLoss class, inherit from nn.Module, and take the tokenizers as arguments
class SequenceLoss(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    #define the forward pass, which takes the y and y_hat tensors as arguments
    def forward(self, y, y_hat):
        #loop through the range of y
        for i in range(len(y)):
            #detokenize the y and y_hat tensors
            y[i] = detokenize(y[i], self.tokenizer)
            y_hat[i] = detokenize(y_hat[i], self.tokenizer)
            print("utils/loss ... tgt_seq: ", y[i])
            print("utils/loss ... pred_seq: ", y_hat[i])
            #calculate the BLEU score for each
            bleu = metrics.bleu_score(y_hat[i], y[i])
            print("utils/loss ... bleu: ", bleu)
        
            
        


        '''
        tính reward
        truyền vào y theo batch_size, 
        tính reward 
        '''