import torch
import torch.nn.functional as F
from torch import nn

from models import MultiwayTransformer
from utils import read_json
from .multiway_transformer import init_weights


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
        self.vqa_vocab = 3128
        self.output = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.LayerNorm(self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.vqa_vocab),
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


    def forward(self, image, quesiton, vqa_targets, train=True):
        text_ids, text_atts = quesiton.input_ids, quesiton.attention_mask
        hidden_state1 = self.backbone(image, None, text_ids, text_atts, mode='vl')
        cls_embedding = hidden_state1[:, self.cls_token_text]
        vqa_logits = self.output(cls_embedding)
        if train:
            nlvr_loss = F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1]
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
            return nlvr_loss, gate_loss
        else:
            return vqa_logits

    