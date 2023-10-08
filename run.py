# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import sys
import time
import random
import argparse
import yaml

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

########### Set it correctly for distributed training across nodes
NNODES = int(os.getenv("ARNOLD_WORKER_NUM"))  # e.g. 1/2/3/4
NPROC_PER_NODE = int(os.getenv("ARNOLD_WORKER_GPU"))  # e.g. 8

MASTER_ADDR = os.getenv("METIS_WORKER_0_HOST")
# MASTER_ADDR = '127.0.0.3'
PORT = os.getenv("METIS_WORKER_0_PORT")
# PORT = '29555'
if ',' in PORT:
    PORT = PORT.split(',')[-1]
MASTER_PORT = int(PORT)
NODE_RANK = int(os.getenv("ARNOLD_ID"))  # e.g. 0/1/2
############

print("NNODES, ", NNODES)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NODE_RANK, ", NODE_RANK)


def get_nnodes(args):  # when using only part of nodes
    if args.dist == 'all':
        return NNODES
    else:
        return 1


def get_dist_launch(args):  # some examples
    if args.dist == 'all':  # use all nodes
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    elif args.dist == '1':
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes=1 ".format(NPROC_PER_NODE)

    elif args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.launch --nproc_per_node=4 " \
               "--nnodes=1 "

    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.launch --nproc_per_node=1 " \
                "--master_port=1234 " \
               "--nnodes=1 ".format(num)

    else:
        raise ValueError


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local

def run_pretrain_beit(args):
    print("### Start pre-training BEiT", flush=True)
    dist_launch = get_dist_launch(args)
    launch_str = f"{dist_launch} --use_env Pretrain_beit.py --config {args.config} --output_dir {args.output_dir}"
    if args.auto_resume:
        launch_str += ' --auto_resume'
    if args.gc:
        launch_str += ' --gc'
    if args.fs:
        launch_str += ' --fs'
    print(launch_str)
    os.system(launch_str)

def run_nlvr2_beit_config():
    dist_launch = get_dist_launch(args)

    # assert os.path.exists("images/nlvr2")

    print("### Training NLVR2_biattn", flush=True)
    print(args.config)
    os.system(f"{dist_launch} "
            f"--use_env NLVR_beit.py --config {args.config} "
            f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} "
            f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} "
            f"{' --evaluate' if args.evaluate else ''} "
            f"{f' --seed {args.seed}' if args.seed else ''} "
            f"{' --gc' if args.gc else ''}")

def run_itr_flickr():
    dist_launch = get_dist_launch(args)

    # assert os.path.exists("images/flickr30k-images")

    print("### Training Retrieval Flickr", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env Retrieval.py --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{' --zeroshot' if args.zeroshot else ''}"
              f"{' --auto_resume' if args.auto_resume else ''}"
              f"{' --gc' if args.gc else ''}" )

def run_itr_coco():
    dist_launch = get_dist_launch(args)

    # assert os.path.exists("images/coco")

    print("### Training Retrieval COCO", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env Retrieval.py --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{' --zeroshot' if args.zeroshot else ''}"
              f"{' --auto_resume' if args.auto_resume else ''}"
              f"{' --gc' if args.gc else ''}" )


def run_vqa(args):
    dist_launch = get_dist_launch(args)

    # assert os.path.exists("images/coco") and os.path.exists("images/visualgenome")

    print("### Training VQA", flush=True)
    if not os.path.exists(args.config): args.config = './configs/VQA.yaml'

    print(f'### Config File: {args.config}')

    launch_str = f"{dist_launch} " + \
                 f"--use_env VQA.py --config {args.config} " + \
                 f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} " + \
                 f"--bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} {'--gc' if args.gc else ''} {f' --seed {args.seed}' if args.seed else ''}"
    
    if args.auto_resume:
        launch_str += ' --auto_resume'
    
    print(launch_str)

    os.system(launch_str)

def run(args):
    if not args.task.startswith('BPretrain'):
        assert hexists(args.checkpoint) or hexists(args.load_ckpt_from)

    if args.task.startswith('BPretrain'):
        run_pretrain_beit(args)
    elif args.task == 'itr_2_tasks':
        output_dir = args.output_dir
        if args.output_hdfs:
            args.output_hdfs = args.output_hdfs + '_coco'
        args.output_dir = output_dir + '_coco'
        args.config = "config_coco.yaml"
        run_itr_coco()
        if args.output_hdfs:
            args.output_hdfs = args.output_hdfs.replace('_coco', '_flickr')
        args.output_dir = output_dir + '_flickr'
        args.config = "config_flickr.yaml"
        run_itr_flickr()
    elif args.task == 'itr_coco':
        output_dir = args.output_dir
        if args.output_hdfs:
            args.output_hdfs = args.output_hdfs + '_coco'
        args.output_dir = output_dir + '_coco'
        args.config = "config_coco.yaml"
        run_itr_coco()
    elif args.task == 'itr_flickr':
        output_dir = args.output_dir
        if args.output_hdfs:
            args.output_hdfs = args.output_hdfs + '_flickr'
        args.output_dir = output_dir + '_flickr'
        args.config = "config_flickr.yaml"
        run_itr_flickr()

    elif args.task == 'vqa':
        run_vqa(args)
    
    elif args.task.startswith('NLVR_vit_lr'):
        if not args.config:
            args.config = 'configs/'+args.task+'.yaml'
        run_nlvr2_beit_config()

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")

    parser.add_argument('--config', default='', type=str, help="if not given, use default")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                           "this option only works for fine-tuning scripts.")

    parser.add_argument('--checkpoint', default='', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default='', type=str, help="load domain pre-trained params")

    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, required=True, help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--output_hdfs', type=str, default='', help="HDFS path required by VQA and Refcoco, "
                                                                    "to collect eval results among nodes")

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--zeroshot', action='store_true', help="zeroshot on downstream tasks")
    parser.add_argument('--gc', action='store_true', help='gradient_checkpointing')
    parser.add_argument('--auto_resume', action='store_true', help='auto resume ')
    parser.add_argument('--beit', action='store_true', help='whether use beit or xvlm')
    parser.add_argument('--fs', action='store_true')
    parser.add_argument('--vlmo_config', type=str, default='')
    parser.add_argument('--vlmo_ckpt', type=str, default='')
    parser.add_argument('--augmentation', type=str, default='')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_mult', type=float)
    parser.add_argument('--warmup', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--mitc_ratio', type=float)
    parser.add_argument('--mitm_ratio', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--k_test', type=int)
    parser.add_argument('--decoder_depth', type=int)
    parser.add_argument('--load_head', action='store_true')
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--biattn', action='store_true')
    parser.add_argument('--mitc_detach', action='store_true')
    parser.add_argument('--seed', type=int)
    # parser.add_argument('--vocab_size', type=str, default='32k')

    args = parser.parse_args()

    if MASTER_ADDR == 'SET_IT':
        print("### warning: the settings for distributed training is not filled (ignore this if you only use one node)")

    if '/SET/PATH/TO/hadoop/bin/hdfs' in HADOOP_BIN:
        print("### warning: you have not set the path to hadoop_bin (ignore this if you don't use HDFS)")

    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)

    if len(args.output_hdfs):
        assert hexists(os.path.dirname(args.output_hdfs))

    if len(args.config):
        assert hexists(args.config)

        if args.config.startswith('hdfs://'):
            args.config = get_from_hdfs(args.config)

    if args.checkpoint.startswith('hdfs://'):
        args.checkpoint = get_from_hdfs(args.checkpoint)
    if not args.task.startswith('itr'):
        if args.config:
            config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        else:
            config = yaml.load(open('configs/'+args.task+'.yaml', 'r'), Loader=yaml.Loader)
        if args.vlmo_config:
            config['vlmo_config'] = 'configs/' + args.vlmo_config
        if args.vlmo_ckpt:
            config['vlmo_ckpt'] = args.vlmo_ckpt
        if args.mitc_ratio:
            config['mitc_ratio'] = args.mitc_ratio
        if args.mitm_ratio:
            config['mitm_ratio'] = args.mitm_ratio
        if args.mitc_detach:
            config['mitc_detach'] = True
        if args.augmentation:
            assert args.augmentation in ['', 'color', 'rand']
            if args.augmentation == '':
                config['augmentation'] = None
            else:
                config['augmentation'] = args.augmentation
        if args.lr:
            config['optimizer']['lr'] = args.lr
            config['schedular']['lr'] = args.lr
        if args.lr_mult:
            config['optimizer']['lr_mult'] = args.lr_mult
        if args.weight_decay:
            config['optimizer']['weight_decay'] = args.weight_decay
        if args.warmup:
            config['schedular']['num_warmup_steps'] = args.warmup
        if args.epoch:
            config['schedular']['epochs'] = args.epoch
        if args.decoder_depth:
            config['decoder_depth'] = args.decoder_depth
        if args.biattn:
            config['biattn'] = True
        if args.load_head:
            config['load_head'] = True
        # if args.single_ffn:
        #     config['single_ffn'] = True
        if '32k' in config.get('vlmo_config', ''):
            config['text_encoder'] = 'data/bert-base-uncased'
        elif '64k' in config.get('vlmo_config', ''):
            config['text_encoder'] = 'data/beit_tokenizer'
        elif 'beit3' in config.get('vlmo_config', ''):
            config['text_encoder'] = 'data/beit3_tokenizer'
        else:
            assert not args.config.startswith('BP')
        yaml.dump(config, open('config.yaml', 'w'))
        args.config = 'config.yaml'
    else:
        config_coco = yaml.load(open("configs/Retrieval_coco_ft.yaml", 'r'), Loader=yaml.Loader)
        config_flickr = yaml.load(open("configs/Retrieval_flickr_ft.yaml", 'r'), Loader=yaml.Loader)
        if args.vlmo_config: 
            config_coco['vlmo_config'] = 'configs/' + args.vlmo_config
            config_flickr['vlmo_config'] = 'configs/' + args.vlmo_config
        if args.augmentation:
            assert args.augmentation in ['', 'color', 'rand']
            if args.augmentation == '':
                config_coco['augmentation'] = None
                config_flickr['augmentation'] = None
            else:
                config_coco['augmentation'] = args.augmentation
                config_flickr['augmentation'] = args.augmentation
        if args.lr:
            config_coco['optimizer']['lr'] = args.lr
            config_flickr['optimizer']['lr'] = args.lr
            config_coco['schedular']['lr'] = args.lr
            config_flickr['schedular']['lr'] = args.lr
        if args.epoch:
            config_coco['schedular']['epochs'] = args.epoch
            config_flickr['schedular']['epochs'] = args.epoch
        if args.lr_mult:
            config_coco['optimizer']['lr_mult'] = args.lr_mult
            config_flickr['optimizer']['lr_mult'] = args.lr_mult
        if args.weight_decay:
            config_coco['optimizer']['weight_decay'] = args.weight_decay
            config_flickr['optimizer']['weight_decay'] = args.weight_decay
        if args.warmup:
            config_coco['schedular']['num_warmup_steps'] = args.warmup
            config_flickr['schedular']['num_warmup_steps'] = args.warmup
        if args.load_head:
            config_coco['load_head'] = True
            config_flickr['load_head'] = True
        if args.rerank:
            config_coco['rerank'] = True
            config_flickr['rerank'] = True
        if args.k_test:
            config_coco['k_test'] = args.k_test
            config_flickr['k_test'] = args.k_test
        if '32k' in config_coco['vlmo_config']:
            config_coco['text_encoder'] = 'data/bert-base-uncased'
            config_flickr['text_encoder'] = 'data/bert-base-uncased'
        elif '64k' in config_coco['vlmo_config']:
            config_coco['text_encoder'] = 'data/beit_tokenizer'
            config_flickr['text_encoder'] = 'data/beit_tokenizer'
        elif 'beit3' in config_coco.get('vlmo_config', ''):
            config_coco['text_encoder'] = 'data/beit3_tokenizer'
            config_flickr['text_encoder'] = 'data/beit3_tokenizer'
        else:
            raise NotImplementedError
        yaml.dump(config_coco, open('config_coco.yaml', 'w'))
        yaml.dump(config_flickr, open('config_flickr.yaml', 'w'))

    run(args)

