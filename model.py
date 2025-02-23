import math
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import timm

from channel import *
from model_util import *
from functools import partial
from trans_decoder import Decoder
from transformers import BertModel
from transformers import MobileViTFeatureExtractor
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from typing import List, Callable, Union, Any, TypeVar, Tuple
from model_util import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from model_util import PositionalEncoding, ViTEncoder_imgcr, SPTEncoder, ViTEncoder_vqa, ViTEncoder_msa
from base_args import IMGC_NUMCLASS, TEXTC_NUMCLASS, IMGR_LENGTH, TEXTR_NUMCLASS, VQA_NUMCLASS, MSA_NUMCLASS, \
    PATCH_SIZE, BERT_SIZE

from base_args import NUM_UE, SHARE_RATIO
from unet_model import UNet

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'TDeepSC_imgc_model',
    'TDeepSC_textc_model']


class TDeepSC_imgc(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0,
                 encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=12, decoder_num_classes=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, num_classes=0, **kwargs
                 ):
        super().__init__()
        self.net = timm.create_model("vit_base_patch" + str(PATCH_SIZE) + "_" + str(img_size), pretrained=True)
        self.net.head = nn.Linear(self.net.head.in_features, 10)

        self.num_symbol = kwargs['n_sym_img']

        self.encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbol)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(self.num_symbol, encoder_embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, ta_perform=None,
                test_snr=torch.FloatTensor([12])):

        if self.training:
            # noise_snr, noise_std = noise_gen(self.training)
            # noise_std, noise_snr = noise_std.cuda(), noise_snr.cpu().item()
            noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)
        else:
            noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)


        x = self.net.patch_embed(img)
        cls_token = self.net.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.net.pos_drop(x + self.net.pos_embed)
        x = self.net.blocks(x)
        x = self.net.norm(x)

        x = self.encoder_to_channel(x)
        x = power_norm_batchwise(x.unsqueeze(1))
        x = self.channel.Rayleigh(x, noise_std.item())
        x = self.channel_to_decoder(x)

        x = self.net.pre_logits(x[:, 0, 0])
        # x = self.head(x.mean(1))
        x = self.net.head(x)

        return x

# class TransformerMapper(nn.Module): 
#     def __init__(self, input_dim, output_dim, num_layers=2, 
#                  num_heads=4, dim_feedforward=256, dropout=0.1):
#         """
#         Args:
#             input_dim: Dimension of the conditioning vector.
#             output_dim: Desired output feature dimension.
#             num_tokens: Number of learned query tokens (for sequence output).
#         """
#         super().__init__()
#         # self.token_embeddings = nn.Parameter(torch.randn(num_tokens, input_dim))
#         decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, 
#                                                    dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(input_dim, output_dim)
    
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (batch, seq_len, input_dim)
#         Returns:
#             Output tensor of shape (batch, output_dim) if num_tokens==1, 
#             or (batch, num_tokens, output_dim) if num_tokens > 1.
#         """
#         query = x.transpose(0,1)
#         memory = x.transpose(0,1)
#         out = self.transformer_decoder(query, memory)
#         out = out.transpose(0,1)
#         out = self.fc(out)
#         return out
           
class TDeepSC_imgr(nn.Module):
    def __init__(self,
                 img_size=224, 
                 encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=12, decoder_num_classes=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, num_classes=0, **kwargs
                 ):
        super().__init__()
        
        # self.img_encoder = ViTEncoder_imgcr(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans,
        #                         num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
        #                         num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate,
        #                         drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
        #                         use_learnable_pos_emb=use_learnable_pos_emb)      
        self.net = UNet(num_ue=NUM_UE, share_ratio=SHARE_RATIO, num_symbols=kwargs['n_sym_img'], 
                        in_channels=3, depth=4, merge_mode='concat')
        # self.net = timm.create_model("vit_small_patch" + str(PATCH_SIZE) + "_" + str(img_size) + ".dino", pretrained=True)
        # for param in self.net.parameters():
        #     param.requires_grad = False
        # self.net.head = nn.Linear(decoder_embed_dim, IMGR_LENGTH)

        # self.num_symbol = kwargs['n_sym_img']
        # encoder_embed_dim = 384 
        # decoder_embed_dim = 1024
        
        # self.encoder_to_channel = nn.Sequential(
        #     nn.Linear(encoder_embed_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.num_symbol))
        
        # self.channel_to_decoder = nn.Sequential(
        #     nn.Linear(self.num_symbol, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64*14*14)
        # )
        # self.decoder = nn.Sequential(
        #     # (64, 14, 14) -> (32, 28, 28)
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     # (32, 28, 28) -> (16, 56, 56)
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     # (16, 56, 56) -> (8, 112, 112)
        #     nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     # (8, 112, 112) -> (output_channels, 224, 224)
        #     nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
        #     # Use Tanh if your images are normalized to [-1,1], or Sigmoid for [0,1]
        #     nn.Sigmoid()
        # )

        # if SHARE_RATIO != 0:
        #     self.channel_to_share = nn.Conv2d(in_channels=NUM_UE, out_channels=1, 
        #                                   kernel_size=(1,int(1/SHARE_RATIO)), stride=(1,int(1/SHARE_RATIO)))
        # else:
        #     self.channel_to_share =  nn.Linear(self.num_symbol, int(self.num_symbol * SHARE_RATIO))

        # self.channel_to_private = nn.Linear(self.num_symbol, int(self.num_symbol * (1 - SHARE_RATIO)))
        # self.channel = Channels()

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    # def get_num_layers(self):
    #     return len(self.blocks)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        # if self.training:
        #     noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)
        # else:
        #     noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)
        
        x = self.net(img)
        return x
        # x = self.encoder_to_channel(x)
        # # print(f"Encoder2Channel: {x.shape}") # (128, 32)

        # if SHARE_RATIO != 0:
        #     batch_size, num_sym = x.shape
        #     shared = torch.reshape(x, [batch_size // NUM_UE, NUM_UE, num_sym]) # (32, 4, 32)
        #     # shared = shared.reshape(shared.shape[0] * shared.shape[2], NUM_UE, -1) # (32* 49, 4, 32) 
        #     shared = shared.unsqueeze(2) # (32, 4, 1, 32)
        #     shared = self.channel_to_share(shared) # (32, 1, 1, 16)
        #     shared = shared.reshape(batch_size // NUM_UE, int(num_sym*SHARE_RATIO)) # (32, 16)
        #     shared = shared.repeat(NUM_UE, 1) # (128, 16)
        #     # shared = shared.repeat_interleave(NUM_UE, dim=0)
        # else:   
        #     shared = self.channel_to_share(x)
        # # print(f"Shared: {shared.shape}")
        
        # private = self.channel_to_private(x) # (128, 16)
        # # print(f"Private: {private.shape}")
        
        # x = torch.cat([shared,private], dim=1) # (128, 32)
        # x = power_norm_batchwise(x)
        
        # x = self.channel_to_decoder(x)
        # x = x.reshape(batch_size, 64, 14, 14)
        # x = self.decoder(x)

        # return x

class TDeepSC_vqa(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, decoder_num_classes=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, num_classes=0, **kwargs
                 ):
        super().__init__()
        self.img_encoder = ViTEncoder_vqa(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans,
                                          num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,
                                          depth=encoder_depth,
                                          num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                          drop_rate=drop_rate,
                                          drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
                                          use_learnable_pos_emb=use_learnable_pos_emb)

        bert_ckpt = f"./pretrain_models/bert-{BERT_SIZE}/"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        if BERT_SIZE == 'tiny':
            encoder_dim_text = 128
        elif BERT_SIZE == 'small':
            encoder_dim_text = 512
        else:
            encoder_dim_text = 512

        self.num_symbols_img = kwargs['n_sym_img']  # Keep all feature vectors
        self.num_symbols_text = kwargs['n_sym_text']  # Keep all feature vectors

        self.img_encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbols_img)
        self.text_encoder_to_channel = nn.Linear(encoder_dim_text, self.num_symbols_text)
        self.img_channel_to_decoder = nn.Linear(self.num_symbols_img, decoder_embed_dim)
        self.text_channel_to_decoder = nn.Linear(self.num_symbols_text, decoder_embed_dim)

        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(25, decoder_embed_dim)

        self.channel = Channels()
        self.head = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std, noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)
        x_img = self.img_encoder(img, ta_perform)
        batch_size = x_img.shape[0]
        x_text = self.text_encoder(input_ids=text, return_dict=False)[0]
        x_img = self.img_encoder_to_channel(x_img)
        x_text = self.text_encoder_to_channel(x_text)
        x_img = power_norm_batchwise(x_img[:, 0:3])
        x_text = power_norm_batchwise(x_text[:, 0:2])

        x_img = self.channel.Rayleigh(x_img, noise_std.item())
        x_text = self.channel.Rayleigh(x_text, noise_std.item())
        x_img = self.img_channel_to_decoder(x_img)
        x_text = self.text_channel_to_decoder(x_text)

        x = torch.cat([x_img, x_text], dim=1)

        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(query_embed, x, None, None, None)
        x = self.head(x.mean(1))
        x = torch.sigmoid(x)
        return x


class TDeepSC_textc(nn.Module):
    def __init__(self,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., decoder_num_classes=384,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, init_values=0., **kwargs
                 ):
        super().__init__()
        bert_ckpt = f"./pretrain_models/bert-{BERT_SIZE}/"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        if BERT_SIZE == 'tiny':
            encoder_embed_dim = 128
        elif BERT_SIZE == 'small':
            encoder_embed_dim = 512
        else:
            encoder_embed_dim = 512

        self.num_symbols = kwargs['n_sym_text']

        self.encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbols)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(self.num_symbols, decoder_embed_dim)

        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(25, decoder_embed_dim)

        self.head = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        x = self.text_encoder(input_ids=text, return_dict=False)[0]
        batch_size = x.shape[0]
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std, noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)
        x = self.encoder_to_channel(x[:, 0])
        x = power_norm_batchwise(x)
        x = self.channel.Rayleigh(x, noise_std.item())
        x = self.channel_to_decoder(x)

        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(query_embed, x, None, None, None)

        x = self.head(x.mean(1))
        return x


class TDeepSC_textr(nn.Module):
    def __init__(self,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., decoder_num_classes=384,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, init_values=0., **kwargs
                 ):
        super().__init__()
        bert_ckpt = f"./pretrain_models/bert-{BERT_SIZE}/"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        if BERT_SIZE == 'tiny':
            encoder_embed_dim = 128
        elif BERT_SIZE == 'small':
            encoder_embed_dim = 512
        else:
            encoder_embed_dim = 512

        self.num_symbols = kwargs['n_sym_text']
        print(self.num_symbols)

        self.encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbols)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(self.num_symbols, decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)

        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(66, decoder_embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        x = self.text_encoder(input_ids=text, return_dict=False)[0]
        batch_size = x.shape[0]
        if self.training:
            noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)
        else:
            noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)
        x = self.encoder_to_channel(x)
        x = power_norm_batchwise(x)
        x = self.channel.Rayleigh(x, noise_std.item())
        x = self.channel_to_decoder(x)

        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(x, x, None, None, None)

        x = self.head(x[:, 0:, ])
        return x


class TDeepSC_msa(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, decoder_num_classes=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, num_classes=0, **kwargs
                 ):
        super().__init__()
        self.img_encoder = ViTEncoder_msa(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans,
                                          num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,
                                          depth=encoder_depth,
                                          num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                          drop_rate=drop_rate,
                                          drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
                                          use_learnable_pos_emb=use_learnable_pos_emb)

        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans, num_classes=encoder_num_classes, embed_dim=128,
                                      depth=encoder_depth, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, drop_rate=drop_rate,
                                      drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
                                      use_learnable_pos_emb=use_learnable_pos_emb)

        bert_ckpt = f"./pretrain_models/bert-{BERT_SIZE}/"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        if BERT_SIZE == 'tiny':
            encoder_dim_text = 128
        elif BERT_SIZE == 'small':
            encoder_dim_text = 512
        else:
            encoder_dim_text = 512

        self.num_symbols_img = kwargs['n_sum_img']
        self.num_symbols_text = kwargs['n_sum_text']
        self.num_symbols_spe = kwargs['n_sum_spe']

        self.img_encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbols_img)
        self.text_encoder_to_channel = nn.Linear(encoder_dim_text, self.num_symbols_text)
        self.spe_encoder_to_channel = nn.Linear(128, self.num_symbols_spe)

        self.channel = Channels()

        self.img_channel_to_decoder = nn.Linear(self.num_symbols_img, decoder_embed_dim)
        self.text_channel_to_decoder = nn.Linear(self.num_symbols_text, decoder_embed_dim)
        self.spe_channel_to_decoder = nn.Linear(self.num_symbols_spe, decoder_embed_dim)

        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(25, decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, speech=None, ta_perform=None, test_snr=torch.FloatTensor([-2])):
        x_text = self.text_encoder(input_ids=text, return_dict=False)[0]
        x_img = self.img_encoder(img, ta_perform)
        x_spe = self.spe_encoder(speech, ta_perform)

        batch_size = x_img.shape[0]

        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std, noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10 ** (-test_snr / 20)

        x_img = self.img_encoder_to_channel(x_img)
        x_text = self.text_encoder_to_channel(x_text)
        x_spe = self.spe_encoder_to_channel(x_spe)

        x_img = power_norm_batchwise(x_img[:, 0].unsqueeze(1))
        x_text = power_norm_batchwise(x_text[:, 0].unsqueeze(1))
        x_spe = power_norm_batchwise(x_spe[:, 0].unsqueeze(1))

        x_img = self.channel.Rayleigh(x_img, noise_std.item())
        x_text = self.channel.Rayleigh(x_text, noise_std.item())
        x_spe = self.channel.Rayleigh(x_spe, noise_std.item())

        x_img = self.img_channel_to_decoder(x_img)
        x_text = self.text_channel_to_decoder(x_text)
        x_spe = self.spe_channel_to_decoder(x_spe)

        x = torch.cat([x_text, x_img, x_spe], dim=1)
        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(query_embed, x, None, None, None)
        x = self.head(x.mean(1))
        return x


@register_model
def TDeepSC_imgc_model(pretrained=False, **kwargs):
    model = TDeepSC_imgc(
        img_size=224,
        patch_size=PATCH_SIZE,
        encoder_embed_dim=768,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def TDeepSC_imgr_model(pretrained=False, **kwargs):
    model = TDeepSC_imgr(
        img_size=224,
        patch_size=PATCH_SIZE,
        encoder_embed_dim=768,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def TDeepSC_textc_model(pretrained=False, **kwargs):
    model = TDeepSC_textc(
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def TDeepSC_textr_model(pretrained=False, **kwargs):
    model = TDeepSC_textr(
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def TDeepSC_vqa_model(pretrained=False, **kwargs):
    model = TDeepSC_vqa(
        img_size=224,
        patch_size=PATCH_SIZE,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def TDeepSC_msa_model(pretrained=False, **kwargs):
    model = TDeepSC_msa(
        mode='small',
        img_size=224,
        patch_size=PATCH_SIZE,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model