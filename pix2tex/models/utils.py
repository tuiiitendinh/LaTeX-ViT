import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score

from . import hybrid
from . import vit
from . import transformer

def detokenize(tokens, tokenizer):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
    # print("Running detokenize...")
    for b in range(len(toks)):
        for i in reversed(range(len(toks[b]))):
            if toks[b][i] is None:
                toks[b][i] = ''
            toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
            if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                del toks[b][i]
    return toks

# Define a Model class, which is a subclass of the nn.Module class from PyTorch
class Model(nn.Module):
    def __init__(self, encoder, decoder, data, args):
        # Call the parent constructor
        super().__init__()

        # Initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.args = args
        self.dataset = data

    # Define a method for running data parallelism
    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        # If there are no device_ids or only one, run the model without parallelism
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)

        # If no output device is specified, use the first device in device_ids as the output device
        if output_device is None:
            output_device = device_ids[0]

        # Create replicas of the model for each device
        replicas = nn.parallel.replicate(self, device_ids)

        # Distribute the input tensor and kwargs across the devices
        inputs = nn.parallel.scatter(x, device_ids)
        kwargs = nn.parallel.scatter(kwargs, device_ids)

        # Apply the model to the inputs in parallel
        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)

        # Gather the outputs from each device to the output device
        return nn.parallel.gather(outputs, output_device).mean()

    # Define the forward pass
    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor,  **kwargs):
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    # Define a method for generating output sequences without calculating gradients
    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        # Generate a sequence starting with the beginning-of-sequence token
        return self.decoder.generate((torch.LongTensor([self.args.bos_token]*len(x))[:, None]).to(x.device), 
                                     self.args.max_seq_len, eos_token=self.args.eos_token, 
                                     context=self.encoder(x), temperature=temperature)

# Define a function for creating a model
def get_model(data, args):
    # Choose the encoder structure
    if args.encoder_structure.lower() == 'vit':
        encoder = vit.get_encoder(args)
    elif args.encoder_structure.lower() == 'hybrid':
        encoder = hybrid.get_encoder(args)
    else:
        raise NotImplementedError('Encoder structure "%s" not supported.' % args.encoder_structure)

    # Get the decoder
    decoder = transformer.get_decoder(args)

    # Move the encoder and decoder to the specified device
    encoder.to(args.device)
    decoder.to(args.device)

    # Create the model
    model = Model(encoder, decoder, data, args)

    # If using Weights & Biases for experiment tracking, start tracking the model
    if args.wandb:
        import wandb
        wandb.watch(model)

    return model
