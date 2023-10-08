import argparse
import datetime
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import fairscale
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from transformers import AutoTokenizer

import utils
from dataset import create_dataset, create_loader, create_sampler
from models.model_nlvr_beit import BEiT
from optim import create_optimizer
from scheduler import create_scheduler
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hcopy, hexists


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler):
    model.train()
    scaler = GradScaler()
    # scaler = ShardedGradScaler()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    amp_dtype = torch.float16 if 'V100' in torch.cuda.get_device_name() else torch.bfloat16
    amp_dtype = torch.float32
    print('### amp use', amp_dtype)
    for i, (image0, image1, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        # with autocast(dtype=amp_dtype):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        text_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=40, return_tensors="pt").to(device)
        # text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        
        nlvr_loss, gate_loss = model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=True)
        loss = nlvr_loss + gate_loss
        # scaler.scale(loss).backward()
        loss.backward()
        accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
        clip_grad_norm_(model.parameters() , accelerator_clip_grad_norm)
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        scheduler.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.update(gate_loss=gate_loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for image0, image1, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)
        text_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=40, return_tensors="pt").to(device)
        # text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        prediction = model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=False)
 
        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('nlvr', config)
    datasets = [train_dataset, val_dataset, test_dataset]

    train_dataset_size = len(train_dataset)
    train_batch_size = config['batch_size']
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")
        print(f"### test data {len(test_dataset)}", flush=True)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']] * 3,
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])


    print("Creating model")
    model = BEiT(config=config)
    model.load_pretrain(args.checkpoint)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model_without_ddp = model
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model, args.gc)

    if args.gc:
        print('### use fairscale gradient checkpoint')
        model = fairscale.nn.data_parallel.ShardedDataParallel(model, optimizer)
        model = fairscale.nn.checkpoint.checkpoint_wrapper(model)
        # model.backbone._set_gradient_checkpoint(True)
    else:
        print('### use torch ddp')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    tokenizer = AutoTokenizer.from_pretrained(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        val_stats = evaluate(model, val_loader, tokenizer, device)
        test_stats = evaluate(model, test_loader, tokenizer, device)

        if utils.is_main_process():
            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            print(log_stats)

        dist.barrier()

    else:
        print("Start training")
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(train_batch_size*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        checkpointer = Checkpointer(args.output_hdfs if hexists(args.output_hdfs) else args.output_dir)

        max_epoch = config['schedular']['epochs']

        best = 0
        best_epoch = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler)
            val_stats = evaluate(model, val_loader, tokenizer, device)
            test_stats = evaluate(model, test_loader, tokenizer, device)

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

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
                        # 'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch + 1,
                }
                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=epoch,
                                             training_states=latest_save_obj)

                if float(val_stats['acc']) > best:
                    # save_obj = {
                    #     'model': model_without_ddp.state_dict(),
                    #     # 'optimizer': optimizer.state_dict(),
                    #     # 'lr_scheduler': lr_scheduler.state_dict(),
                    #     'config': config,
                    #     # 'epoch': epoch,
                    # }
                    # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = float(val_stats['acc'])
                    best_epoch = epoch

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()

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
    parser.add_argument('--config', default='./configs/NLVR.yaml')
    parser.add_argument('--output_dir', default='output/nlvr')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--load_nlvr_pretrain', action='store_true')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--beit', action='store_true')
    parser.add_argument('--gc', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
