import torch
import torch.nn.functional as F
from torch import nn

from models import MultiwayTransformer
from utils import read_json
from .multiway_transformer import init_weights
from .model_pretrain_beit import AllGather
import torch.distributed as dist

allgather = AllGather.apply


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
        self.num_vision_tokens = self.backbone.num_patch_embed
        self.config = config
        self.itc_type = config.get('itc_type', ['vl', 'fusion']) # vl -> pure vision & language
        itc_proj_dim = config.get('itc_proj_dim', self.hidden_size)
        self.load_head = config.get('load_head', False)
        # self.itc_type = config.get('itc_type', ['vl', 'fusion']) # vl -> pure vision & language
        self.itc_temp = nn.Parameter(torch.ones([]) * 0.07) # hard code temp
        if 'fusion' in self.itc_type:
            print('### add fusion contrastive loss')
            self.itc_vision_proj_head = nn.Linear(self.hidden_size, itc_proj_dim, bias=False)
            self.itc_vision_proj_head.apply(init_weights)
            self.itc_text_proj_head = nn.Linear(self.hidden_size, itc_proj_dim, bias=False)
            self.itc_text_proj_head.apply(init_weights)
        if 'vl' in self.itc_type:
            print('### add pure vision and text contrastive loss')
            self.itc_vl_vision_proj_head = nn.Linear(self.hidden_size, itc_proj_dim, bias=False)
            self.itc_vl_vision_proj_head.apply(init_weights)
            self.itc_vl_text_proj_head = nn.Linear(self.hidden_size, itc_proj_dim, bias=False)
            self.itc_vl_text_proj_head.apply(init_weights)

        self.itm_head = nn.Linear(self.hidden_size, 2)
        self.itm_head.apply(init_weights)
        self.init_params = []
        
    def load_pretrain(self, ckpt_path):
        state_dict = self.backbone.load_pretrain_beit(ckpt_path, self.config)
        if self.load_head:
            if 'mitc_vision_proj_head.weight' in state_dict:
                state_dict['itc_vision_proj_head.weight'] = state_dict['mitc_vision_proj_head.weight']
                state_dict['itc_text_proj_head.weight'] = state_dict['mitc_text_proj_head.weight']
                state_dict['itc_temp'] = state_dict['mitc_temp']
            if 'itc_vl_vision_proj_head.weight' in state_dict:
                state_dict['itc_vision_proj_head.weight'] = state_dict['itc_vl_vision_proj_head.weight']
                state_dict['itc_text_proj_head.weight'] = state_dict['itc_vl_text_proj_head.weight']
                # state_dict['itc_temp'] = state_dict['mitc_temp']
            if 'mitm_head.weight' in state_dict:
                state_dict['itm_head.weight'] = state_dict['mitm_head.weight']
                state_dict['itm_head.bias'] = state_dict['mitm_head.bias']
        else:
            state_dict.pop('itc_vision_proj_head.weight', None)
            state_dict.pop('itc_text_proj_head.weight', None)
            state_dict.pop('itc_temp', None)
            state_dict.pop('itm_head.weight', None)
            state_dict.pop('itm_head.bias', None)
            
        msg = self.load_state_dict(state_dict=state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_path)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)
        self.init_params.extend(msg.missing_keys)

    def forward_itc(self, 
                    image,
                    image_mask,
                    text_ids,
                    text_atts):
        hidden_states_vision, attention_mask_vision = self.backbone.pre_forward(image, image_mask, None, None, 'vision')
        hidden_states_text, attention_mask_text = self.backbone.pre_forward(None, None, text_ids, text_atts, 'text')
        all_hidden_states_vision = []
        # rel pos emb remove tmp
        # if self.backbone.relative_position_embed:
        #     relative_position_bias_list_vision = \
        #         self.backbone.encoder.get_rel_pos_bias(self.backbone.encoder.relative_position_index)
        #     relative_position_bias_list_text = \
        #         self.backbone.encoder.get_rel_pos_bias(self.backbone.encoder.text_relative_position_index)
        #     relative_position_bias_list_vl = \
        #         self.backbone.encoder.get_rel_pos_bias(self.backbone.encoder.text_imag_relative_position_index)
        for idx, encoder_layer in enumerate(self.backbone.encoder.layers): 
            hidden_states_vision = encoder_layer(
                    hidden_states=hidden_states_vision, 
                    attention_mask=attention_mask_vision, 
                    relative_position_bias=None,
                    mode='vision'
            )
            all_hidden_states_vision.append(hidden_states_vision)

        all_hidden_states_text = []
        for idx, encoder_layer in enumerate(self.backbone.encoder.layers): 
            hidden_states_text = encoder_layer(
                    hidden_states=hidden_states_text, 
                    attention_mask=attention_mask_text, 
                    relative_position_bias=None,
                    mode='text'
            )
            all_hidden_states_text.append(hidden_states_text)
        
        # hidden_states_vision_vl = all_hidden_states_vision[min(self.ffn_layers) - 1]
        # hidden_states_text_vl = all_hidden_states_text[min(self.ffn_layers) - 1]
        # for idx, encoder_layer in enumerate(self.backbone.encoder.layers[min(self.ffn_layers):]):
        #     hidden_states_vision_vl = encoder_layer(
        #             hidden_states=hidden_states_vision_vl, 
        #             attention_mask=attention_mask_vision, 
        #             relative_position_bias=None,
        #             mode='vl'
        #     )
        #     hidden_states_text_vl = encoder_layer(
        #             hidden_states=hidden_states_text_vl, 
        #             attention_mask=attention_mask_text, 
        #             relative_position_bias=None,
        #             mode='vl'
        #     )
        hidden_states_vision = self.backbone.norm(all_hidden_states_vision[-1])
        # hidden_states_vision_vl = self.backbone.norm(hidden_states_vision_vl)
        hidden_states_text = self.backbone.norm(all_hidden_states_text[-1])
        # hidden_states_text_vl = self.backbone.norm(hidden_states_text_vl)
        return hidden_states_vision, hidden_states_text

    def cal_itc_loss(self, itc_logits, idx=None):
        bsz = itc_logits.shape[0]
        if idx is None:
            labels = torch.arange(bsz).cuda()
        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == bsz
            pos_idx = torch.eq(idx, idx.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)
        
        loss_i2t = -torch.sum(F.log_softmax(itc_logits, dim=1) * labels, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(itc_logits.t(), dim=1) * labels, dim=1).mean()
        return (loss_i2t + loss_t2i) / 2

    def cal_itm_loss(self, image, text_ids, text_atts, itc_logits, idx=None):
        bs, text_len = text_ids.size()
        itm_labels = torch.cat([torch.ones(bs), torch.zeros(2*bs)], dim=0).long().cuda()
        with torch.no_grad():
            cur_rank = dist.get_rank()
            cur_st, cur_end = cur_rank * bs, (cur_rank + 1) * bs
            image_all = allgather(image, dist.get_rank(), dist.get_world_size())
            text_ids_all = allgather(text_ids, dist.get_rank(), dist.get_world_size())
            text_atts_all = allgather(text_atts, dist.get_rank(), dist.get_world_size())

            weights_i2t = F.softmax(itc_logits.float(), dim=1) + 1e-5
            weights_t2i = F.softmax(itc_logits.t().float(), dim=1) + 1e-5
            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == text_ids_all.size(0)
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)
            
            weights_i2t = weights_i2t[cur_st:cur_end]
            weights_t2i = weights_t2i[cur_st:cur_end]

        images_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            images_neg.append(image_all[neg_idx])
        images_neg = torch.stack(images_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_ids_all[neg_idx])
            text_atts_neg.append(text_atts_all[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # forward once / 3 times
        # itm_image = torch.cat([image, images_neg, image], dim=0)
        # itm_text_ids = torch.cat([text_ids, text_ids, text_ids_neg], dim=0)
        # itm_text_atts = torch.cat([text_atts, text_atts, text_atts_neg], dim=0)
        # print(itm_image.size(), itm_text_ids.size(), itm_text_atts.size())

        # itm_outputs = self.backbone(image=itm_image, image_mask=None, text_ids=itm_text_ids, text_atts=itm_text_atts, mode='vl')
        # itm_score = itm_outputs[:, self.num_vision_tokens+1,:]
        # itm_logits = self.itm_head(itm_score)

        itm_outputs1 = self.backbone(image=image, image_mask=None, text_ids=text_ids, text_atts=text_atts, mode='vl')
        itm_outputs1 = itm_outputs1[:, self.num_vision_tokens+1,:]
        itm_outputs2 = self.backbone(image=images_neg, image_mask=None, text_ids=text_ids, text_atts=text_atts, mode='vl')
        itm_outputs2 = itm_outputs2[:, self.num_vision_tokens+1,:]
        itm_outputs3 = self.backbone(image=image, image_mask=None, text_ids=text_ids_neg, text_atts=text_atts_neg, mode='vl')
        itm_outputs3 = itm_outputs3[:, self.num_vision_tokens+1,:]
        itm_outputs = torch.cat([itm_outputs1, itm_outputs2, itm_outputs3], dim=0)
        itm_logits = self.itm_head(itm_outputs)

        itm_loss = F.cross_entropy(itm_logits, itm_labels)
        # itm_acc = (itm_logits.argmax(1) == itm_labels).float().mean()
        return itm_loss

    def forward_eval(self, image, text_ids, text_atts, mode):
        if mode == 'vision':
            hidden_states_vision, attention_mask_vision = self.backbone.pre_forward(image, None, None, None, 'vision')
            all_hidden_states_vision = []
            for idx, encoder_layer in enumerate(self.backbone.encoder.layers):
                hidden_states_vision = encoder_layer(
                        hidden_states=hidden_states_vision, 
                        attention_mask=attention_mask_vision, 
                        relative_position_bias=None,
                        mode='vision'
                )
            vision_cls_norm = F.normalize(self.itc_vision_proj_head(hidden_states_vision[:,0]), dim=-1)
            return vision_cls_norm
        else:
            hidden_states_text, attention_mask_text = self.backbone.pre_forward(None, None, text_ids, text_atts, 'text')
            all_hidden_states_text = []
            for idx, encoder_layer in enumerate(self.backbone.encoder.layers): 
                hidden_states_text = encoder_layer(
                        hidden_states=hidden_states_text, 
                        attention_mask=attention_mask_text, 
                        relative_position_bias=None,
                        mode='text'
                )
            text_cls_norm = F.normalize(self.itc_text_proj_head(hidden_states_text[:,0]),dim=-1)
            return text_cls_norm

    def forward(self, image, text_ids, text_atts, idx=None):
        with torch.no_grad():
            self.itc_temp.clamp_(0.001, 0.5)
        if idx is not None:
            idx = idx.view(-1, 1)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
        else:
            idx_all = None
        itc_vision_feat, itc_text_feat = self.forward_itc(image, None, text_ids, text_atts)
        if 'fusion' in self.itc_type:
            vision_cls_norm = F.normalize(self.itc_vision_proj_head(itc_vision_feat[:,0]), dim=-1)
            text_cls_norm = F.normalize(self.itc_text_proj_head(itc_text_feat[:,0]),dim=-1)
            image_feat_all = allgather(vision_cls_norm, torch.distributed.get_rank(), torch.distributed.get_world_size())
            text_feat_all = allgather(text_cls_norm, torch.distributed.get_rank(), torch.distributed.get_world_size())
            itc_logits = image_feat_all @ text_feat_all.t() / self.itc_temp
            itc_loss = self.cal_itc_loss(itc_logits, idx=idx_all)

        # if 'vl' in self.itc_type:
        #     vl_vision_cls_norm = F.normalize(self.itc_vl_vision_proj_head(itc_vl_vision_feat[:,0]), dim=-1)
        #     vl_text_cls_norm = F.normalize(self.itc_vl_text_proj_head(itc_vl_text_feat[:,0]),dim=-1)
        #     vl_image_feat_all = allgather(vl_vision_cls_norm, torch.distributed.get_rank(), torch.distributed.get_world_size())
        #     vl_text_feat_all = allgather(vl_text_cls_norm, torch.distributed.get_rank(), torch.distributed.get_world_size())
        #     vl_itc_logits = vl_image_feat_all @ vl_text_feat_all.t() / self.itc_temp
        #     vl_itc_loss = self.cal_itc_loss(vl_itc_logits, idx=idx_all)
        #     itc_loss = (itc_loss + vl_itc_loss) / 2

        itm_loss = self.cal_itm_loss(image, text_ids, text_atts, itc_logits.detach(), idx=idx_all)

        with torch.no_grad():
            gate_loss = itc_loss - itc_loss
        if self.ffn_type == 'moe':
            loss, i = 0., 0
            for layer in self.backbone.encoder.layers:
                if hasattr(layer, 'aux_loss_vl') and layer.aux_loss_vl_i:
                    loss = loss + layer.aux_loss_vl / layer.aux_loss_vl_i
                    i += 1
                    layer.aux_loss_vl = 0
                    layer.aux_loss_vl_i = 0
            gate_loss = self.moe_lossweight * loss / i

        return itc_loss, itm_loss, gate_loss