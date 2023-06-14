import torch
from torchtext.data import metrics
from torch import nn
from pix2tex.eval import detokenize

class SequenceLoss(nn.Module):
    def __init__(self, tokenizer) -> None:
        super(SequenceLoss).__init__()
        self.tokenizer = tokenizer

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        for i in range(len(y)):
            tgt_seq = detokenize(y[i], self.tokenizer)
            pred_seq = detokenize(y_hat[i], self.tokenizer)
            print("tgt_seq: ", tgt_seq)
            print("pred_seq: ", pred_seq)
            print("BLEU score: ", metrics.bleu_score(pred_seq, tgt_seq))
