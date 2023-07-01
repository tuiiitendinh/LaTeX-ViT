import torch
import torch.nn.functional as F
import numpy as np
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p
from x_transformers import TransformerWrapper, Decoder


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    #hàm này trả về 2 giá trị:
    # out: 1D Tensor chứa generated sequence
    # sequence_probs: 2D Tensor chứa xác suất của tất cả token trong generated sequence
    
    def generate(self, start_tokens, seq_len=256, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        # initializes the variable sequence_probs with the same value as the out tensor
        sequence_probs = out #set to the same shape [32, 1]
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1) #[32, 1]
            
            #lấy giá trị xác suất của tất cả token trong phân phối đa thức từ biến sample
            prob_values = probs[np.arange(b), sample.flatten()]
            prob_values = prob_values.reshape(b, 1)

            #concat các giá trị xác suất của từng token -> 2D array
            # -> ppxs của tất cả token trong câu
            sequence_probs = torch.cat((sequence_probs, prob_values), dim=-1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break
        
        out = out[:, t:]
        sequence_probs[:, 1:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        
        return out, sequence_probs


def get_decoder(args):
    return CustomARWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                **args.decoder_args
            )),
        pad_value=args.pad_token)
