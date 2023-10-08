import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from regex import E
from scipy import interpolate
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.models.bert.modeling_bert import (
    BertConfig, BertEmbeddings, BertPredictionHeadTransform)
from transformers.utils import logging

import models.layer as moe_layer
from models.vit import interpolate_pos_embed
from utils import torch_io


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class MultiwayTransformerMLP(nn.Module):
    def __init__(self, hidden_act, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.save_attention = False
    
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
    
    def get_attention_map(self):
        return self.attention_map


    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
        
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        if self.save_attention:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients) 
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiwayTransformerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.save_attention = False
    
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        value_states = value_states.view(*proj_shape)
        src_len = value_states.size(1)
        attn_weights = query_states @ key_states.transpose(-2, -1)
        if relative_position_bias is not None:
            attn_weights = attn_weights + relative_position_bias.unsqueeze(0)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if self.save_attention:
            self.save_attention_map(attn_weights)
            attn_weights.register_hook(self.save_attn_gradients) 

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = attn_weights

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

class MultiwayTransformerEnocderLayer(nn.Module):
    def __init__(self, 
                hidden_size, 
                hidden_act, 
                num_attention_heads, 
                intermediate_size, 
                num_tokens, 
                drop_path, 
                layer_scale_init_values=0.1, 
                use_ffn=False,
                ffn_type='moe',
                ffn_param=None,
                single_ffn=False):
        super().__init__()
        print('### MultiwayTransformerEnocderLayer', drop_path, use_ffn, ffn_type, ffn_param)
        self.attn = Attention(hidden_size, num_attention_heads, qkv_bias=True)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.num_vision_tokens, self.num_text_tokens = num_tokens
        
        self.ffn_param = ffn_param
        self.ffn_type = ffn_type
        self.use_ffn = use_ffn
        self.single_ffn = single_ffn
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        if self.ffn_type =='moe' and self.use_ffn:
            self.norm2_vl = nn.LayerNorm(hidden_size, eps=1e-6)
            mlp_tmp = MultiwayTransformerMLP(hidden_act, hidden_size, intermediate_size)
            num_experts = self.ffn_param.get('num_experts', 8)
            type_route = self.ffn_param.get('type_route', False)
            topk = self.ffn_param.get('topk', 2)
            self.num_experts = num_experts
            self.topk = topk
            if type_route:
                self.mlp_moe = moe_layer.MoE(hidden_size=hidden_size, num_experts=num_experts, k=topk, expert=mlp_tmp, num_tokens=self.num_vision_tokens)
            else:
                self.mlp_moe = moe_layer.MoE(hidden_size=hidden_size, num_experts=num_experts, k=topk, expert=mlp_tmp, num_tokens=None)
        else:
            if self.single_ffn['type'] == 'single':
                self.mlp_imag = MultiwayTransformerMLP(hidden_act, hidden_size, intermediate_size)
                self.norm2_imag = nn.LayerNorm(hidden_size, eps=1e-6)
            elif self.single_ffn['type'] == 'moe':
                self.norm2_imag = nn.LayerNorm(hidden_size, eps=1e-6)
                moe_param = self.single_ffn.get('moe_param',  {"num_experts": 2, "topk": 1, "type_route": True})
                mlp_tmp = MultiwayTransformerMLP(hidden_act, hidden_size, intermediate_size)
                num_experts = moe_param.get('num_experts', 2)
                type_route = moe_param.get('type_route', True)
                topk = moe_param.get('topk', 1)
                self.num_experts = num_experts
                self.topk = topk
                if type_route:
                    self.mlp_moe = moe_layer.MoE(hidden_size=hidden_size, num_experts=num_experts, k=topk, expert=mlp_tmp, num_tokens=self.num_vision_tokens)
                else:
                    self.mlp_moe = moe_layer.MoE(hidden_size=hidden_size, num_experts=num_experts, k=topk, expert=mlp_tmp, num_tokens=None)
            else:
                self.mlp_imag = MultiwayTransformerMLP(hidden_act, hidden_size, intermediate_size)
                self.norm2_imag = nn.LayerNorm(hidden_size, eps=1e-6)
                self.mlp_text = MultiwayTransformerMLP(hidden_act, hidden_size, intermediate_size)
                self.norm2_text = nn.LayerNorm(hidden_size, eps=1e-6)
            if use_ffn:
                self.norm2_vl = nn.LayerNorm(hidden_size, eps=1e-6)
                self.mlp_vl = MultiwayTransformerMLP(hidden_act, hidden_size, intermediate_size)

        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((hidden_size)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((hidden_size)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.aux_loss_vl = 0
        self.aux_loss_vl_i = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: None,
        relative_position_bias = None,
        mode='vl'
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            x=hidden_states,
            mask=attention_mask,
            relative_position_bias=relative_position_bias
        )
        hidden_states = residual + self.drop_path(hidden_states * self.gamma_1)

        residual = hidden_states
        # hidden_states = self.layer_norm2(hidden_states) # note layer norm is independent
        if mode == 'vl':
            if self.use_ffn:
                hidden_states = self.norm2_vl(hidden_states)
                if self.ffn_type == 'moe':
                    hidden_states, aux_loss, _ = self.mlp_moe(hidden_states, mode=mode)
                    self.aux_loss_vl = self.aux_loss_vl + aux_loss
                    self.aux_loss_vl_i += 1
                else:
                    hidden_states = self.mlp_vl(hidden_states)
            else:
                # note [vision, lang]
                if self.single_ffn['type'] == 'single':
                    hidden_states = self.mlp_imag(self.norm2_imag(hidden_states))
                elif self.single_ffn['type'] == 'moe':
                    hidden_states, aux_loss, _ = self.mlp_moe(self.norm2_imag(hidden_states))
                    self.aux_loss_vl = self.aux_loss_vl + aux_loss
                    self.aux_loss_vl_i += 1
                else:
                    hidden_vision = hidden_states[:,:-self.num_text_tokens]
                    hidden_lang = hidden_states[:,-self.num_text_tokens:]
                    hidden_vision = self.mlp_imag(self.norm2_imag(hidden_vision))
                    hidden_lang = self.mlp_text(self.norm2_text(hidden_lang))
                    hidden_states = torch.cat([hidden_vision, hidden_lang], dim=1)
            
        elif mode == 'vision':
            if self.use_ffn and self.ffn_type == 'moe':
                hidden_states = self.norm2_vl(hidden_states)
                hidden_states, aux_loss, _ = self.mlp_moe(hidden_states, mode=mode)
                self.aux_loss_vl = self.aux_loss_vl + aux_loss
                self.aux_loss_vl_i += 1
            else:
                if self.single_ffn['type'] == 'single':
                    hidden_states = self.mlp_imag(self.norm2_imag(hidden_states))
                elif self.single_ffn['type'] == 'moe':
                    hidden_states, aux_loss, _ = self.mlp_moe(self.norm2_imag(hidden_states))
                    self.aux_loss_vl = self.aux_loss_vl + aux_loss
                    self.aux_loss_vl_i += 1
                else:
                    hidden_states = self.norm2_imag(hidden_states)
                    hidden_states = self.mlp_imag(hidden_states)
        elif mode == 'text':
            if self.use_ffn and self.ffn_type == 'moe':
                hidden_states = self.norm2_vl(hidden_states)
                hidden_states, aux_loss, _ = self.mlp_moe(hidden_states, mode=mode)
                self.aux_loss_vl = self.aux_loss_vl + aux_loss
                self.aux_loss_vl_i += 1
            else:
                if self.single_ffn['type'] == 'single':
                    hidden_states = self.mlp_imag(self.norm2_imag(hidden_states))
                elif self.single_ffn['type'] == 'moe':
                    hidden_states, aux_loss, _ = self.mlp_moe(self.norm2_imag(hidden_states))
                    self.aux_loss_vl = self.aux_loss_vl + aux_loss
                    self.aux_loss_vl_i += 1
                else:
                    hidden_states = self.norm2_text(hidden_states)
                    hidden_states = self.mlp_text(hidden_states)
        else:
            print(f'unsupport mode {mode}')
            raise NotImplementedError
        
        hidden_states = residual + self.drop_path(hidden_states * self.gamma_2)

        return hidden_states

class MultiwayTransformerEncoder(nn.Module):

    def __init__(self, 
                 hidden_size, 
                 hidden_act, 
                 num_attention_heads, 
                 intermediate_size, 
                 num_hidden_layers, 
                 num_tokens, 
                 drop_path_rate, 
                 ffn_type='moe',
                 ffn_layers=[7,9,11],
                 ffn_param=None,
                 single_ffn=False):
        print(f'### MultiwayTransformerEncoder {ffn_type} in {ffn_layers}')
        super().__init__()
        self.depth = num_hidden_layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.layers = nn.ModuleList([
                MultiwayTransformerEnocderLayer(
                    hidden_size=hidden_size,
                    hidden_act=hidden_act, 
                    num_attention_heads=num_attention_heads, 
                    intermediate_size=intermediate_size,
                    num_tokens=num_tokens, 
                    drop_path=dpr[_],
                    use_ffn= _ in ffn_layers,
                    ffn_type=ffn_type,
                    ffn_param=ffn_param,
                    single_ffn=single_ffn
                )
                for _ in range(num_hidden_layers)
        ])
        self.relative_position_embed = False
        self.num_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.gradient_checkpointing = False
        self.init_std = 0.02 # note! hard code tmp
        self.apply(self._init_weights)
    
    def get_rel_pos_bias(self, relative_position_index):
        if self.relative_position_embed:
            relative_position_bias = F.embedding(relative_position_index.long().to(self.relative_position_bias_table.device),
                                                    self.relative_position_bias_table)
            all_relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, x, y
            relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)
            return relative_position_bias_list
        else:
            return [None] * self.num_layers

    def _build_relative_position_embed(self, config, image_size, patch_size):
        self.relative_position_embed = True
        window_size = (int(image_size / patch_size), int(image_size / patch_size)) #(14, 14)
        num_heads = self.num_attention_heads
        max_text_len_of_initckpt = config['texts']['max_tokens'] if 'texts' in config else 196 #196
        max_text_len = config['pairs']['max_tokens'] if 'pairs' in config else 40 #40
        max_imag_len = window_size[0] * window_size[1] + 1 #197
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        self.all_num_relative_distance = self.num_relative_distance + self.text_num_relative_distance + 2

        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.all_num_relative_distance, num_heads * self.num_layers))
        trunc_normal_(self.relative_position_bias_table, std=self.init_std)
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = relative_position_index
        
        text_position_ids = torch.arange(max_text_len-1)
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2-max_text_len_of_initckpt) #-194
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len, ) * 2, dtype=relative_coords.dtype)
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        # self.text_relative_position_index = text_relative_position_index
        
        text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (self.num_relative_distance)
        imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (self.num_relative_distance + 1)

        text_row_relative_position_index = torch.cat((text2imag_relative_position_index, text_relative_position_index), 1)
        imag_row_relative_position_index = torch.cat((relative_position_index, imag2text_relative_position_index), 1)
        text_imag_relative_position_index = torch.cat((imag_row_relative_position_index, text_row_relative_position_index), 0)
        self.text_imag_relative_position_index = text_imag_relative_position_index

        text_position_ids = torch.arange(max_text_len_of_initckpt-1)
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2-max_text_len_of_initckpt) #-194
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len_of_initckpt, ) * 2, dtype=relative_coords.dtype)
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.text_relative_position_index = text_relative_position_index

        print('vision relative pos index:', self.relative_position_index.size())
        print('text relative pos index:', self.text_relative_position_index.size())
        print('vision_text relative pos index:', self.text_imag_relative_position_index.size())

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer, 'mlp_imag'):
                rescale(layer.mlp_imag.fc2.weight.data, layer_id + 1)
            if hasattr(layer, 'mlp_text'):
                rescale(layer.mlp_text.fc2.weight.data, layer_id + 1)
            if hasattr(layer, 'mlp_vl'):
                rescale(layer.mlp_vl.fc2.weight.data, layer_id + 1)
            if hasattr(layer, 'mlp_moe'):
                for i, expert in enumerate(layer.mlp_moe.moe.experts.experts):
                    rescale(expert.fc2.weight.data, layer_id + 1)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    
    def forward(self, inputs_embeds, attention_mask, mode='vl'):
        hidden_states = inputs_embeds
        bs, token_len, hidden_dim = inputs_embeds.size()
        if self.relative_position_embed:
            if mode == 'vision':
                relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)
            elif mode == 'text':
                relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)
            else:
                relative_position_bias_list = self.get_rel_pos_bias(self.text_imag_relative_position_index)
        for idx, encoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, mode)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    relative_position_bias_list[idx] if self.relative_position_embed else None
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states=hidden_states, 
                    attention_mask=attention_mask, 
                    relative_position_bias=relative_position_bias_list[idx] if self.relative_position_embed else None,
                    mode=mode
                )

        return hidden_states

class MultiwayTransformer(nn.Module):
    def __init__(self, 
                image_size,
                patch_size,
                hidden_size, 
                hidden_act, 
                num_attention_heads, 
                intermediate_size, 
                num_hidden_layers,
                vocab_size,
                relative_position_embed=False,
                drop_path_rate=0,
                config=None,
                ffn_type='moe',
                ffn_layers=[7,9,11],
                ffn_param=None,
                single_ffn=False):
        super().__init__()
        self.init_std = 0.02 # note! hard code tmp
        self.num_attention_heads = num_attention_heads

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.num_patch_embed = (self.image_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=hidden_size, kernel_size=self.patch_size, stride=self.patch_size, bias=True
        )
        trunc_normal_(self.patch_embed.weight, std=self.init_std)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

        # vision embeddings
        self.class_embedding_vision = nn.Parameter(torch.zeros(hidden_size))
        self.num_pos_embed_vision = self.num_patch_embed + 1
        self.type_embed_vision = nn.Parameter(torch.zeros(hidden_size))
        # init vision
        trunc_normal_(self.class_embedding_vision, std=self.init_std)
        trunc_normal_(self.type_embed_vision, std=self.init_std)

        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.max_text_length = config['texts']['max_tokens'] if 'texts' in config else 196
        self.text_length_pair = config['pairs']['max_tokens'] if 'pairs' in config else 40
        self.num_pos_embed_text = self.max_text_length
        self.type_embed_text = nn.Parameter(torch.zeros(hidden_size))
        self.layer_norm_text = nn.LayerNorm(hidden_size, eps=1e-6)
        self.drop_out_text = nn.Dropout(p=0.1)
        # init text
        self.word_embedding.apply(init_weights)
        trunc_normal_(self.type_embed_text, std=self.init_std)

        self.pre_layrnorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.ffn_type = ffn_type
        self.encoder = MultiwayTransformerEncoder(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_tokens=(self.num_pos_embed_vision, self.text_length_pair),
            drop_path_rate=drop_path_rate,
            ffn_type=ffn_type,
            ffn_layers=ffn_layers,
            ffn_param=ffn_param,
            single_ffn=single_ffn
        )
        
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.num_tokens_all = self.num_pos_embed_text + self.num_pos_embed_vision
        self.relative_position_embed = relative_position_embed
        if self.relative_position_embed:
            print('### use relative position embed')
            self.encoder._build_relative_position_embed(config=config, image_size=self.image_size, patch_size=self.patch_size)
        else:
            print('### use absolute position embed')
            self.pos_embed_vision = nn.Parameter(torch.zeros(self.num_pos_embed_vision, hidden_size))
            trunc_normal_(self.pos_embed_vision, std=self.init_std)
            self.register_buffer("position_ids_vision", torch.arange(self.num_pos_embed_vision).expand((1, -1)))

            self.pos_embed_text = nn.Parameter(torch.zeros(self.num_pos_embed_text, hidden_size))
            trunc_normal_(self.pos_embed_text, std=self.init_std)
            self.register_buffer("position_ids_text", torch.arange(self.num_pos_embed_text))
    
    def gen_mask_embed_vision(self):
        self.mask_embed_vision = nn.Parameter(torch.zeros(self.hidden_size))
        trunc_normal_(self.mask_embed_vision, std=self.init_std)
        
    def load_pretrain_beit(self, ckpt_path, config):
        checkpoint = torch_io.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        num_patches = (config['image_res'] // config['patch_size']) ** 2
        if 'backbone.pos_embed_vision' in state_dict:
            pos_embed_vision_new = interpolate_pos_embed(state_dict['backbone.pos_embed_vision'].unsqueeze(0), num_patches, 1)
            state_dict['backbone.pos_embed_vision'] = pos_embed_vision_new.squeeze(0)
        state_dict.pop('backbone.position_ids_vision', None)
        if 'backbone.encoder.relative_position_bias_table' in state_dict and self.relative_position_embed:
            rel_pos_bias = state_dict["backbone.encoder.relative_position_bias_table"]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = self.encoder.relative_position_bias_table.size()
            print('origin :', rel_pos_bias.size(), 'target :', self.encoder.relative_position_bias_table.size())
            dst_patch_shape = (self.image_size // self.patch_size, ) * 2
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                state_dict.pop("backbone.encoder.relative_position_index", None)
                state_dict.pop("backbone.encoder.text_relative_position_index", None)
                state_dict.pop("backbone.encoder.text_imag_relative_position_index", None)
                
                print("Position interpolate from %dx%d to %dx%d" % (
                    src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions = %s" % str(x))
                print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict["backbone.encoder.relative_position_bias_table"] = new_rel_pos_bias
            print(state_dict["backbone.encoder.relative_position_bias_table"].size())
            assert state_dict["backbone.encoder.relative_position_bias_table"].size() == self.encoder.relative_position_bias_table.size()
        return state_dict
    
    def _set_gradient_checkpoint(self, gradient_checkpointing=False):
        self.encoder.gradient_checkpointing = gradient_checkpointing

    def _gen_hidden_states_vision(self, image, image_mask, ids_keep=None):
        batch_size = image.size()[0]
        patch_embeds = self.patch_embed(image)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 256, 196, 768
        if ids_keep is not None:
            hidden_states_vision = patch_embeds + self.pos_embed_vision[1:]
            hidden_states_vision = torch.gather(patch_embeds, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.hidden_size))
            class_embeds_vision = self.class_embedding_vision.expand(batch_size, 1, -1) + self.pos_embed_vision[:1]
            embeddings_vision = torch.cat([class_embeds_vision, hidden_states_vision], dim=1)
            hidden_states_vision = embeddings_vision + self.type_embed_vision
            attention_mask = torch.ones(hidden_states_vision.size()[:2]).cuda()
            return hidden_states_vision, attention_mask
        if image_mask is not None:
            patch_embeds[image_mask] = self.mask_embed_vision.to(patch_embeds.dtype)
        class_embeds_vision = self.class_embedding_vision.expand(batch_size, 1, -1)
        embeddings_vision = torch.cat([class_embeds_vision, patch_embeds], dim=1)
        if not self.relative_position_embed:
            hidden_states_vision = embeddings_vision + self.pos_embed_vision + self.type_embed_vision
        else:
            hidden_states_vision = embeddings_vision + self.type_embed_vision
        
        attention_mask = torch.ones((batch_size, self.num_pos_embed_vision)).cuda()

        return hidden_states_vision, attention_mask
    
    def _gen_hidden_state_text(self, text_ids, text_atts, mode='vl'):
        batch_size, text_len = text_ids.size()
        # origin text emebedding
        embeddings_text = self.word_embedding(text_ids)
        embeddings_text = self.layer_norm_text(embeddings_text)
        embeddings_text = self.drop_out_text(embeddings_text)
        if not self.relative_position_embed:
            hidden_states_text = embeddings_text + self.pos_embed_text[:text_len] + self.type_embed_text
        else:
            hidden_states_text = embeddings_text + self.type_embed_text

        return hidden_states_text, text_atts

    def pre_forward(self, image, image_mask, text_ids, text_atts, mode='vl', vision_ids_keep=None):
        hidden_states, attention_mask = None, None
        if mode == 'vl':
            hidden_states = []
            attention_mask = []
            if image is not None:
                hidden_states_vision, attention_mask_vision = self._gen_hidden_states_vision(image, image_mask, ids_keep=vision_ids_keep)
                hidden_states.append(hidden_states_vision)
                attention_mask.append(attention_mask_vision)
            if text_ids is not None:
                hidden_states_text, attention_mask_text = self._gen_hidden_state_text(text_ids, text_atts, mode=mode)
                hidden_states.append(hidden_states_text)
                attention_mask.append(attention_mask_text)
            hidden_states = torch.cat(hidden_states, dim=1)
            attention_mask = torch.cat(attention_mask, dim=1)

        elif mode == 'vision':
            hidden_states, attention_mask = self._gen_hidden_states_vision(image, image_mask, ids_keep=vision_ids_keep)

        elif mode == 'text':
            hidden_states, attention_mask = self._gen_hidden_state_text(text_ids, text_atts, mode=mode)
        
        hidden_states = self.pre_layrnorm(hidden_states)
        return hidden_states, attention_mask
        
    def forward(self, 
                image,
                image_mask,
                text_ids,
                text_atts,
                mode='vl',
                vision_ids_keep=None):
        # batch_size = image.size()[0] if image is not None else text_ids.size()[0]
        hidden_states, attention_mask = self.pre_forward(image, image_mask, text_ids, text_atts, mode, vision_ids_keep)
        # token_length = hidden_states.size()[1]
        # attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        # attention_mask = attention_mask.expand((batch_size, 1, token_length, token_length))
        outputs = self.encoder(hidden_states, attention_mask, mode)
        outputs = self.norm(outputs)

        return outputs