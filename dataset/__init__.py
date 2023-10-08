import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from dataset.coco_karpathy_dataset import (coco_karpathy_caption_eval,
                                           coco_karpathy_train,
                                           coco_karpathy_train_scst)
from dataset.nlvr_dataset import nlvr_dataset
from dataset.pretrain_dataset import ImageTextJsonLocalDataset
from dataset.randaugment import RandomAugment
from dataset.re_dataset import re_eval_dataset, re_train_dataset
from dataset.vqa_dataset import vqa_dataset


def common_collate_fn(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        elif isinstance(x[0], np.ndarray):
            batch_tensors.append(torch.tensor(np.array(x), dtype=torch.long))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors

def create_dataset(dataset, config):
    # normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                    interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    vlmo_common_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                    interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4)
    ])

    naive_pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                    interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ])
    if config.get('mim_type','tokenizer') == 'pixel':
        common_transform = naive_pretrain_transform
    else:
        common_transform = vlmo_common_transform 
    # print('### pretrain transform use', common_transform)
    if 'dalle' in config.get('vision_tokenizer', ''):
        tokenizer_transform = transforms.Compose([
            transforms.Resize(config['image_res']//2, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
    else:
        tokenizer_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    vlmo_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    weak_pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0),
                                    interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    if config.get('augmentation', None) == None:
        # print('### Downstream Without Augmentation')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                        interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif config.get('augmentation', None) == 'color':
        # print('### Downstream use ColorJitter Augmentation')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                        interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # print('### Downstream use RandomAugment Augmentation')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                        interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    # print('### downstream augmentation', train_transform)

    train_transform_wohflip = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                    interpolation=Image.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    if dataset == 'pretrain_beit':
        if 'transform' in config.keys():
            if config['transform']=='naive':
                trans = naive_pretrain_transform
                print('NAIVE TRANSFORM')
            elif config['transform']=='weak':
                trans = weak_pretrain_transform
                print('WEAK TRANSFORM')
            else:
                trans = pretrain_transform
        else:
            trans = pretrain_transform
        general_dataset = ConcatDataset(
                [ImageTextJsonLocalDataset(config, path, transform=(common_transform, tokenizer_transform, vlmo_transform)) 
                for path in config['train_file']]
            )
        return general_dataset
    elif dataset == 're':
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'vqa':
        train_dataset = vqa_dataset(config['train_file'], train_transform_wohflip, config['vqa_root'], config['vg_root'],
                                    split='train', text_encoder=config['text_encoder'], use_roberta=config['use_roberta'], answer_list=config['answer_list'])
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'], text_encoder=config['text_encoder'], use_roberta=config['use_roberta'])
        return train_dataset, vqa_test_dataset
    elif dataset == 'nlvr':
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")


def vqa_collate_fn(batch):
    image_list, question_list, target_list = [], [], []
    for image, question, target in batch:
        image_list.append(image)
        question_list.append(question)
        target_list.append(target)
    return torch.stack(image_list, dim=0), question_list, torch.stack(target_list, dim=0)


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
