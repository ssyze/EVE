import os
import json
import random
import torch
from random import random as rand

from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question

from torchvision.transforms.functional import hflip

from transformers import AutoTokenizer

def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0

class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, split="train", max_ques_words=30, answer_list='',
                 text_encoder='', use_roberta=False):

        self.careful_hflip = True

        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        print('### vqa transform', transform)
        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        print('### vqa vqa_root', self.vqa_root)
        print('### vqa vg_root', self.vg_root)
        self.max_ques_words = max_ques_words

        tokenizer = AutoTokenizer.from_pretrained(text_encoder)


        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = '</s>' if use_roberta else '[SEP]'

        self.answer_list = json.load(open(answer_list, 'r'))
        self.vqa_vocab = 3128
        self.target_index = {}
        for answer in self.answer_list:
            self.target_index[answer] = len(self.target_index)
        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
        
    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def __getitem__(self, index):

        ann = self.ann[index]

        if 'dataset' in ann.keys():
            if ann['dataset'] == 'vqa':
                image_path = os.path.join(self.vqa_root, ann['image'])
            elif ann['dataset'] == 'vg':
                image_path = os.path.join(self.vg_root, ann['image'])
            elif ann['dataset'] == 'gqa':
                image_path = ann['image']
            else:
                raise NotImplementedError

        else:
            image_path = os.path.join(self.vqa_root, ann['image'])

        image = Image.open(image_path).convert('RGB')

        if (self.split != 'test') and rand() < 0.5:
            if self.careful_hflip and self.left_or_right_in(ann['question'], ann['answer']):
                pass
            else:
                image = hflip(image)

        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['question_id']
            return image, question, question_id

        elif self.split == 'train':
            question = pre_question(ann['question'], self.max_ques_words)

            if ('dataset' in ann.keys()) and (ann['dataset'] == 'vg'):
                answer_weight = {ann['answer']: 1.0}

            else:
                answer_weight = {}
                for answer in ann['answer']:
                    answer_weight[answer] = answer_weight.get(answer, 0) + 1
                for label, num in answer_weight.items():
                    answer_weight[label] = get_score(num)
            targets = torch.zeros(self.vqa_vocab)
            for label, score in answer_weight.items():
                targets[self.target_index[label]] = score

            return image, question, targets

        else:
            raise NotImplementedError
