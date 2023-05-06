import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame, to_2tuple
from einops import repeat

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

class PatchEmbeding(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
#         torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
#         torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, patch_size=patch_size, *args, **kwargs)
        self.height, self.width = img_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbeding(img_size=img_size, patch_size=patch_size, in_chans = 1, embed_dim = 256, flatten=False)
        self.pos_encoding = PositionalEncoding2D(channels=256)  # Changed from self.pos_embed to self.pos_encoding

    def forward_features(self, x):
        B, c, h, w = x.shape
        # print("\ninput shape: ", x.shape)
        
        x = self.patch_embed(x).permute(0, 2, 3, 1) #expected embed_dim = 256
        # print("\npatch embeded shape: ", x.shape)
        # x.shape now is torch.Size([1, 24, 2, 256]) as [Batch_size, height, width, embed_dim]

        #we have to expand the shape of cls_token to [Batch, cls_token, width, embed_dim], for example: [1, 1, 2, 256]
        cls_tokens = self.cls_token.expand(B, 1, x.shape[2] , -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print("\ncls_tokens shape: ", cls_tokens.shape)
        
        #concatenate cls_tokens and x
        x = torch.cat((cls_tokens, x), dim = 1)
        # print("\nshape after patch_embed + cls_token: ", x.shape) # torch.Size([1, 25, 2, 256])

        # Add position encoding
        x += self.pos_encoding(x)
        # print("\nshape after pos embedding: ", x.shape) # torch.Size([1, 25, 2, 256])

        x = self.pos_drop(x)
        # print("\nshape after pos drop: ", x.shape) # torch.Size([1, 25, 2, 256])

        # Flatten height and width dimensions to [batch, height*width, embed_dim]
        x = x.reshape(B, -1, x.shape[-1])
        # print("\nshape after flattening height * width: ", x.shape)  # torch.Size([1, 50, 256])

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


def get_encoder(args):
    backbone = ResNetV2(
        layers=args.backbone_layers, num_classes=0, global_pool='', in_chans=args.channels,
        preact=False, stem_type='same', conv_layer=StdConv2dSame)
    min_patch_size = 2**(len(args.backbone_layers)+1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps//min_patch_size, backbone=backbone)

    encoder = CustomVisionTransformer(img_size=(args.max_height, args.max_width),
                                      patch_size=args.patch_size,
                                      in_chans=args.channels,
                                      num_classes=0,
                                      embed_dim=args.dim,
                                      depth=args.encoder_depth,
                                      num_heads=args.heads,
                                      embed_layer=embed_layer
                                      )
    return encoder