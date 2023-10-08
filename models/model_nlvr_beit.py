import torch
import torch.nn.functional as F
from torch import nn

from models import MultiwayTransformer
from utils import read_json
from .multiway_transformer import init_weights

from .attention import MultiheadAttention


class BEiT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        vlmo_config = read_json(config['vlmo_config'])
        print('### MultiwayTransformer config :', vlmo_config)
        self.hidden_size = vlmo_config['vision_width']
        self.text_vocab_size = vlmo_config['vocab_size']
        self.vision_vocab_size = vlmo_config['vision_vocab_size']
        self.ffn_type = vlmo_config.get('ffn_type', 'moe')
        self.moe_lossweight = config.get('gateloss_weight', 0.001)
        self.single_ffn = vlmo_config.get('single_ffn', {"type": "vl"})
        if self.ffn_type == 'moe':
            self.ffn_layers = vlmo_config.get('moeffn_layers', [7,9,11])
            self.ffn_param = vlmo_config.get('moe_param', {"num_experts": 8,"topk": 2})
        elif self.ffn_type == 'vl':
            self.ffn_layers = vlmo_config.get('vlffn_layers', [10,11])
            self.ffn_param = None
        else:
            raise NotImplementedError
        self.backbone = MultiwayTransformer(image_size=config['image_res'], 
                            patch_size=vlmo_config['patch_size'],
                            hidden_size=self.hidden_size,
                            hidden_act=vlmo_config['hidden_act'],
                            num_attention_heads=vlmo_config['num_attention_heads'],
                            intermediate_size=vlmo_config['intermediate_size'],
                            num_hidden_layers=vlmo_config['num_hidden_layers'],
                            vocab_size=self.text_vocab_size,
                            relative_position_embed=vlmo_config.get('relative_position_embed', False),
                            drop_path_rate=vlmo_config.get('drop_path_rate', 0),
                            config=config,
                            ffn_type=self.ffn_type,
                            ffn_layers=self.ffn_layers,
                            ffn_param=self.ffn_param,
                            single_ffn=self.single_ffn)
        self.config = config
        self.biattn = config.get('biattn', False)
        if self.biattn:
            num_attention_heads = vlmo_config['num_attention_heads']
            attention_probs_dropout_prob = 0.1
            self.attn1 = MultiheadAttention(self.hidden_size,
                                            num_attention_heads,
                                            attention_probs_dropout_prob)
            self.attn2 = MultiheadAttention(self.hidden_size,
                                            num_attention_heads,
                                            attention_probs_dropout_prob)
            self.fc = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1))
            self.attn_pool = AttentionPool(self.hidden_size,
                                        attention_probs_dropout_prob)
            self.nlvr2_output = nn.Linear(2*self.hidden_size, 2)
        else:
            self.output = nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                    nn.LayerNorm(self.hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_size * 2, 2),
                )
            self.output.apply(init_weights)
        self.cls_token_vision = 0
        self.cls_token_text = self.backbone.num_pos_embed_vision
        self.init_params = []
        
    def load_pretrain(self, ckpt_path):
        state_dict = self.backbone.load_pretrain_beit(ckpt_path, self.config)
        msg = self.load_state_dict(state_dict=state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_path)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)
        self.init_params.extend(msg.missing_keys)

    def forward(self, image, text_ids, text_atts, targets, train=True):
        image1, image2 = torch.split(image, targets.size()[0])
        if self.biattn:
            hidden_states1, attention_mask1 = self.backbone.pre_forward(image1, None, text_ids, text_atts, mode='vl')
            outputs1 = self.backbone.encoder(hidden_states1, attention_mask1, mode='vl')
            left_out = self.backbone.norm(outputs1)
            hidden_states2, attention_mask2 = self.backbone.pre_forward(image2, None, text_ids, text_atts, mode='vl')
            outputs2 = self.backbone.encoder(hidden_states2, attention_mask2, mode='vl')
            right_out = self.backbone.norm(outputs2)
            left_mask = (attention_mask1 == 0)
            right_mask = (attention_mask2 == 0)
            left_out = left_out.transpose(0, 1)
            right_out = right_out.transpose(0, 1)
            l2r_attn, _ = self.attn1(left_out, right_out, right_out,
                                    key_padding_mask=right_mask)
            r2l_attn, _ = self.attn2(right_out, left_out, left_out,
                                    key_padding_mask=left_mask)
            left_out = self.fc(torch.cat([l2r_attn, left_out], dim=-1)
                            ).transpose(0, 1)
            right_out = self.fc(torch.cat([r2l_attn, right_out], dim=-1)
                                ).transpose(0, 1)
            # attention pooling and final prediction
            left_out = self.attn_pool(left_out, left_mask)
            right_out = self.attn_pool(right_out, right_mask)
            prediction = self.nlvr2_output(
                torch.cat([left_out, right_out], dim=-1))
        else:
            hidden_state1 = self.backbone(image1, None, text_ids, text_atts, mode='vl')
            hidden_state2 = self.backbone(image2, None, text_ids, text_atts, mode='vl')
            cls_embedding = torch.cat(
                    [hidden_state1[:, self.cls_token_text],
                    hidden_state2[:, self.cls_token_text]],
                    dim=1)
            prediction = self.output(cls_embedding)
        nlvr_loss = F.cross_entropy(prediction, targets)
        with torch.no_grad():
            gate_loss = nlvr_loss - nlvr_loss
        if self.ffn_type == 'moe':
            loss, i = 0., 0
            for layer in self.backbone.encoder.layers:
                if hasattr(layer, 'aux_loss_vl') and layer.aux_loss_vl_i:
                    loss = loss + layer.aux_loss_vl / layer.aux_loss_vl_i
                    i += 1
                    layer.aux_loss_vl = 0
                    layer.aux_loss_vl_i = 0
            gate_loss = self.moe_lossweight * loss / i
        if train:
            return nlvr_loss, gate_loss
        else:
            return prediction

class AttentionPool(nn.Module):
    """ attention pooling layer """
    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output