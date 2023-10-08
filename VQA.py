import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer
from transformers import AutoTokenizer
from torch.nn.utils.clip_grad import clip_grad_norm_
import fairscale

import utils
from dataset import (create_dataset, create_loader, create_sampler,
                     vqa_collate_fn)
from dataset.utils import collect_result
from models.model_vqa_beit import BEiT
from optim import create_optimizer
from scheduler import create_scheduler
from utils import torch_io
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hexists, hmkdir


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

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, checkpointer):
    ckpt_frequent_step = config.get('ckpt_frequent_step', 1e9)
    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    skip_steps = config.get('skip_steps', 0)

    for i, (image, question, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header, skip_steps=skip_steps)):
        optimizer.zero_grad()
        cur_step = i + skip_steps + 1
        image, targets = image.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        # answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        # loss = model(image, question_input, answer_input, train=True, k=n, weights=weights)
        loss, gate_loss = model(image, question_input, targets, train=True)
        
        
        loss.backward()
        accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
        clip_grad_norm_(model.parameters() , accelerator_clip_grad_norm)
        optimizer.step()

        scheduler.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(gate_loss=gate_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if utils.is_main_process():
            if cur_step % ckpt_frequent_step == 0:
                latest_save_obj = {
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': scheduler.state_dict(),
                                'config': config,
                                'epoch': epoch,
                                'step': cur_step
                }
                checkpointer.save_checkpoint(model_state=None,
                                            epoch=epoch,
                                            training_states=latest_save_obj,
                                            step=cur_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []

    # answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    # answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        # question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        probs = model(image, question_input, None, train=False)
        
        for ques_id, topk_prob in zip(question_id, probs):
            ques_id = int(ques_id.item())          
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[pred]})   

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if world_size > 8:
        assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

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
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']

    print("Creating vqa datasets")
    train_dataset, vqa_test_dataset = create_dataset('vqa', config)
    datasets = [train_dataset, vqa_test_dataset]

    train_dataset_size = len(train_dataset)
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    print("Creating model")
    tokenizer = AutoTokenizer.from_pretrained(config['text_encoder'])

    print("### pad_token_id, ", train_dataset.pad_token_id)
    print("### eos_token, ", train_dataset.eos_token)
    config['pad_token_id'] = train_dataset.pad_token_id
    config['eos'] = train_dataset.eos_token
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

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)

    if args.evaluate:
        print("Start evaluating")
        vqa_result = evaluation(model, test_loader, tokenizer, device, config)
        result = collect_result(vqa_result, 'vqa_eval', local_wdir=args.result_dir,
                                hdfs_wdir=args.output_hdfs,
                                write_to_hdfs=world_size > 2, save_result=True)
    
        dist.barrier()

    else:
        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model, args.gc)
        if auto_resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)
        
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))



        if args.gc:
            print('### use fairscale checkpoint')
            model = fairscale.nn.data_parallel.ShardedDataParallel(model, optimizer)
            # model.backbone._set_gradient_checkpoint(True)
            model = fairscale.nn.checkpoint.checkpoint_wrapper(model)
        else:
            print('### use torch ddp')
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        
        if auto_resume:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']
            config['skip_steps'] = checkpoint.get('step', 0)
            if config['skip_steps']:
                print(f'skip: {config["skip_steps"]}')

        checkpointer = Checkpointer(args.output_hdfs if hexists(args.output_hdfs) else args.output_dir)

        # if args.distributed:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        for epoch in range(start_epoch, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config, checkpointer)
            config['skip_steps'] = 0

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    # 'epoch': epoch,
                }
                latest_save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch + 1,
                }
                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=epoch,
                                             training_states=latest_save_obj)

            if epoch >= config['start_eval']:
                vqa_result = evaluation(model, test_loader, tokenizer, device, config)
                result = collect_result(vqa_result, 'vqa_result_epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                         write_to_hdfs=world_size > 2, save_result=True)

            dist.barrier()

        os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--gc', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    print(args.output_dir)
    hmkdir(args.output_dir)
    print(args.result_dir)
    hmkdir(args.result_dir)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    main(args, config)
