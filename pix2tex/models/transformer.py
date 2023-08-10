import torch
import torch.nn.functional as F
import numpy as np
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p
from x_transformers import TransformerWrapper, Decoder


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()    
    def generate(self, start_tokens, seq_len=256, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        #initialize sample variable
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens 

        out_20 = start_tokens.repeat_interleave(20, dim=0)


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

            sample_1 = torch.multinomial(probs, 1)

            sample_20 = torch.multinomial(probs, 20)
            sample_20 = sample_20.reshape(-1, 1)
          
            out = torch.cat((out, sample_1), dim=-1)
            out_20 = torch.cat((out_20, sample_20), dim=-1)


            mask = F.pad(mask, (0, 1), value=True)
            #create mask for out_20
            mask_20 = F.pad(mask, (0, 20), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break
         
        #remove start tokens
        out = out[:, t:]
        out_20 = out_20[:, t:]
        
        if num_dims == 1:
            out = out.squeeze(0)
            out_20 = out_20.squeeze(0)

        self.net.train(was_training)
        
        #out: index tokens trong vocab của generated sequences,
        return out, out_20


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