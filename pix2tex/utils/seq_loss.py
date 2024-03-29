import torch
import numpy as np
from torchtext.data import metrics
from torch import nn
from pix2tex.eval import detokenize
from pix2tex.utils import alternatives

'''
    y: labels
    y_hat: prediction
    log_prob_gradients: loss tính được từ data_parallel (sử dụng cross_entropy)
'''

class SequenceLoss(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, y, y_hat, log_prob_gradients):
        y = detokenize(y, self.tokenizer) #detokenize cho label
        y_hat = detokenize(y_hat, self.tokenizer) #detokenize cho predict

        #bleu score cho 2 chuỗi label/predict là điểm reward 
        reward = metrics.bleu_score(y_hat, [alternatives(x) for x in y])
        # print(f"Reward Score: {reward}")

        # Convert reward to a 1D tensor
        reward_tensor = torch.tensor([reward], device=log_prob_gradients.device)

        # Ensure log_prob_gradients is a 1D tensor
        if log_prob_gradients.dim() == 0:
            log_prob_gradients = log_prob_gradients.unsqueeze(0)

        # Compute the dot product
        dot_product = torch.dot(reward_tensor, log_prob_gradients)

        # Return the dot product as the final loss
        return dot_product


'''
    ### Class cũ có sử dụng reinforcement có multinomial 

class SequenceLoss(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def forward(self, y, y_hat, y_tilde, loss):
        y = detokenize(y, self.tokenizer)
        y_hat = detokenize(y_hat, self.tokenizer)
        y_tilde = detokenize(y_tilde, self.tokenizer)
        
        #sampling y_tilde to get the same shape with groundtruth
        # y & y_hat has the shape of [batch_size, seq_len]
        #with batch_size = 32, y_tilde has the shape of [640, seq_len]
        #so we need to sample 20 elements from y_tilde to get the shape of [32, seq_len]
        i, j= 0, 20
        y_tilde_samples = []
        while j<=len(y_tilde):
            y_tilde_samples.append(random.choice(y_tilde[i:j]))
            i+=20
            j+=20
        i, j = 0, 20
    
        # Calculate the BLEU score
        reward = metrics.bleu_score(y_tilde_samples, [alternatives(x) for x in y])
        
        #calculate the reward score for y_tilde, store the scores in a list to calculate average
        score_lst = []
        i, j, t = 0, 20, 0  
        while j <= len(y_tilde):
            tmp = []
            for _ in range(i,j):
                # tmp.append(metrics.bleu_score([y_tilde[_]], [[x] for x in y_hat[t]])) #Assertion Error: the length of 2 sequence is mismatch
                tmp.append(metrics.bleu_score([y_tilde[_]], [y[t]]))
            score_lst.append(tmp)
            i += 20
            j += 20
        score_lst = np.array(score_lst)
        # print(score_lst.shape)

        # print("Reward BLEU list: ", score_lst) #expected 20 elements in this array

        #calculate the average reward score
        r_bar = torch.mean(torch.tensor(score_lst))
        # print("--- r_bar score: ", r_bar)

        res = loss * (reward - r_bar)
        print(f"{loss} * {reward} - {r_bar} = {res}")
        return res
'''