import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Optimizer
from transformers import AutoTokenizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from fairscale.optim.grad_scaler import ShardedGradScaler
import fairscale

import utils
from dataset import create_dataset, create_loader, create_sampler
from models.model_retrieval_beit import BEiT
from optim import create_optimizer
from scheduler import create_scheduler
from utils import torch_io
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hcopy, hexists, hmkdir


def reinit_scheduler_properties_mysched(optimizer: Optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    issue: https://github.com/pytorch/pytorch/issues/27595
    issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
    """
    args = cfg

    if scheduler.optimizer == optimizer:
        # from transformers import get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        scheduler.__init__(optimizer, lr_lambda, last_epoch=-1)

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, scaler):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('acc_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_gate', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('acc_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    amp_dtype = torch.float16 if 'V100' in torch.cuda.get_device_name() else torch.bfloat16
    for i, (image, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        with autocast(dtype=amp_dtype):
            loss_itc, loss_itm, gate_loss = model(image, text_input.input_ids, text_input.attention_mask, idx=idx)
            loss = loss_itc + loss_itm + gate_loss
        
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        # loss.backward()

        accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), accelerator_clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itm=loss_itm.item())
        # metric_logger.update(acc_itm=itm_acc.item())
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_gate=gate_loss.item())
        # metric_logger.update(acc_itc=itc_acc.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, rerank=False):
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = config['batch_size_test_text']  # 256
    text_embeds = []
    all_text_ids = []
    all_text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        text_embed = model.forward_eval(None, text_input.input_ids, text_input.attention_mask, mode='text')
        
        text_embeds.append(text_embed)
        all_text_ids.append(text_input.input_ids)
        all_text_atts.append(text_input.attention_mask)
    
    all_text_ids = torch.cat(all_text_ids, dim=0)
    all_text_atts = torch.cat(all_text_atts, dim=0)

    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    all_images = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed = model.forward_eval(image, None, None, mode='vision')

        image_embeds.append(image_embed)
        all_images.append(image)
    image_embeds = torch.cat(image_embeds, dim=0)
    all_images = torch.cat(all_images, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    
    if rerank:
        k_test = config['k_test']
        print(f'Evaluation with Rerank {k_test}')

        score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

        num_tasks = utils.get_world_size()
        rank = utils.get_rank()
        step = sims_matrix.size(0) // num_tasks
        rest = sims_matrix.size(0) % num_tasks
        steps = [step + 1 if i < rest else step for i in range(num_tasks)]
        start = sum(steps[:rank])
        end = start + steps[rank]
        print(start, end, steps[rank])

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
            cur_images = all_images[start+i].repeat(k_test, 1, 1, 1)
            cur_text_ids = all_text_ids[topk_idx]
            cur_text_atts = all_text_atts[topk_idx]

            rerank_output = model.backbone(image=cur_images, image_mask=None, text_ids=cur_text_ids, text_atts=cur_text_atts, mode='vl')
            rerank_embed = rerank_output[:, model.num_vision_tokens+1,:]
            rerank_scores = model.itm_head(rerank_embed)[:, 1]
            score_matrix_i2t[start+i,topk_idx] = rerank_scores
        
        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
        
        step = sims_matrix.size(0) // num_tasks
        rest = sims_matrix.size(0) % num_tasks
        steps = [step + 1 if i < rest else step for i in range(num_tasks)]
        start = sum(steps[:rank])
        end = start + steps[rank]
        print(start, end, steps[rank])

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
            cur_images = all_images[topk_idx]
            cur_text_ids = all_text_ids[start+i].repeat(k_test, 1) 
            cur_text_atts = all_text_atts[start+i].repeat(k_test, 1) 

            rerank_output = model.backbone(image=cur_images, image_mask=None, text_ids=cur_text_ids, text_atts=cur_text_atts, mode='vl')
            rerank_embed = rerank_output[:, model.num_vision_tokens+1,:]
            rerank_scores = model.itm_head(rerank_embed)[:, 1]
            score_matrix_t2i[start+i,topk_idx] = rerank_scores
        
        if args.distributed:
            dist.barrier()   
            torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
            torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)   
    else:
        score_matrix_i2t = sims_matrix
        score_matrix_t2i = sims_matrix.t()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size
    
    auto_resume = False
    if args.auto_resume:
        ckpt_path = os.path.join(args.output_hdfs, 'training_state_latest.th')
        print(f'starting resume from {ckpt_path}')
        if hexists(ckpt_path):
            auto_resume = True
            checkpoint = torch_io.load(ckpt_path, map_location='cpu')

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating retrieval dataset", flush=True)
    train_dataset, val_dataset, test_dataset = create_dataset('re', config)

    train_dataset_size = len(train_dataset)

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")
        print(f"### data {len(val_dataset)}, batch size, {config['batch_size_test']} x {world_size}")
        print(f"### data {len(test_dataset)}, batch size, {config['batch_size_test']} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])


    print("Creating model", flush=True)
    model = BEiT(config=config)
    model.load_pretrain(args.checkpoint)
    if auto_resume:
        print("Load Resume model:")
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_path)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    tokenizer = AutoTokenizer.from_pretrained(config['text_encoder'])

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    if args.evaluate:
        print("Start evaluating", flush=True)
        score_test_i2t, score_test_t2i = evaluation(model, test_loader, tokenizer, device, config)
        score_test_i2t_rank, score_test_t2i_rank = evaluation(model, test_loader, tokenizer, device, config, rerank=True)

        if utils.is_main_process():
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)

            log_stats = {**{f'test_{k}': v for k, v in test_result.items()}}
            print(log_stats)

            print('-'*30+'rerank result'+'-'*30)

            test_result_rerank = itm_eval(score_test_i2t_rank, score_test_t2i_rank, test_loader.dataset.txt2img, test_loader.dataset.img2txt)

            log_stats_rerank = {**{f'test_rerank_{k}': v for k, v in test_result_rerank.items()}}
            print(log_stats_rerank)

        dist.barrier()

    else:
        print("Start training", flush=True)
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model, args.gc)
        if auto_resume:
            optimizer.load_state_dict(checkpoint['optimizer'])

        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(config['batch_size_train']*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        start_epoch = 0
        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0

        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        scaler = None
        if args.gc:
            print('### use fairscale gradient checkpoint')
            model = fairscale.nn.data_parallel.ShardedDataParallel(model, optimizer)
            model = fairscale.nn.checkpoint.checkpoint_wrapper(model)
            scaler = ShardedGradScaler()
            # model.backbone._set_gradient_checkpoint(True)
        else:
            print('### use torch ddp')
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            scaler = GradScaler()

        if auto_resume:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            scaler.load_state_dict(checkpoint['scaler'])

        model_without_ddp = model.module
        reinit_scheduler_properties_mysched(optimizer, lr_scheduler, arg_sche)

        if not hexists(args.output_hdfs):
            hmkdir(args.output_hdfs)
        checkpointer = Checkpointer(args.output_hdfs if hexists(args.output_hdfs) else args.output_dir)

        # if args.distributed:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        for epoch in range(start_epoch, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config, scaler)

            score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

            # if args.gc:
            #     optimizer.consolidate_state_dict(recipient_rank=-1)
            
            if utils.is_main_process():
                val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
                print('val_result',val_result)
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                print('test_result',test_result)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch}

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if test_result['r_mean'] > best:
                    best = test_result['r_mean']
                    best_epoch = epoch

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    # 'epoch': epoch,
                }
                if args.gc:
                    latest_save_obj = {
                        'config': config,
                        'epoch': epoch,
                    }
                else:
                    latest_save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'scaler': scaler.state_dict()

                    }
                checkpointer.save_checkpoint(model_state=save_obj,
                                                epoch=epoch,
                                                training_states=latest_save_obj)
            
            if config.get('rerank', False) or epoch == max_epoch - 1:
                # score_val_i2t_rank, score_val_t2i_rank, = evaluation(model_without_ddp, val_loader, tokenizer, device, config, rerank=True)
                score_test_i2t_rank, score_test_t2i_rank = evaluation(model_without_ddp, test_loader, tokenizer, device, config, rerank=True)
                
                if utils.is_main_process():
                    # val_result_rerank = itm_eval(score_val_i2t_rank, score_val_t2i_rank, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
                    # print('val_result_rerank',val_result_rerank)
                    test_result_rerank = itm_eval(score_test_i2t_rank, score_test_t2i_rank, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                    print('test_result_rerank',test_result_rerank)

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            #  **{f'val_rerank_{k}': v for k, v in val_result_rerank.items()},
                             **{f'test_rerank_{k}': v for k, v in test_result_rerank.items()},
                             'epoch': epoch}
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    if test_result_rerank['r_mean'] > best:
                        best = test_result_rerank['r_mean']
                        best_epoch = epoch
                
                # torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")
            hcopy(args.output_dir, args.output_hdfs)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")
    parser.add_argument('--output_dir', type=str, required=True)  # this script works for both mscoco and flickr30k
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--zeroshot', action='store_true')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--gc', action='store_true')


    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
