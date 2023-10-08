# EVE: Efficient Vision-Language Pre-training with Masked Prediction and Modality-Aware MoE

[EVE: Efficient Vision-Language Pre-training with Masked Prediction and Modality-Aware MoE](https://arxiv.org/abs/2308.11971)

Release preprint in [arxiv](https://arxiv.org/abs/2308.11971).

Release pre-trained weight and fine-tuning code.

Pre-training code is coming soon.

# Pre-trained and Fine-tuned Models

## Pre-trained Checkpoints
[Base Model with 4M Images and 10M Image-Text Pairs](https://1drv.ms/u/s!Aj0m85zoxqZSgc104mN1Kqms_-dMcw?e=bhasUN).

[Large Model with 4M Images and 10M Image-Text Pairs](https://1drv.ms/u/s!Aj0m85zoxqZSgc19SGJ4hPSWFN_vtw?e=xLBEnv).

[Large Model with 16M Images and 21M Image-Text Pairs](https://1drv.ms/u/s!Aj0m85zoxqZSgc4B3Etw0XzCn-5jMw?e=WjFbyw).

## Fine-tuned Models
[Base Model with 4M Images and 10M Image-Text Pairs](https://1drv.ms/f/s!Aj0m85zoxqZSgc4E3R1kZy3hTz76zg?e=lHww37).

[Large Model with 4M Images and 10M Image-Text Pairs](https://1drv.ms/f/s!Aj0m85zoxqZSgc4FYrUnTIdg24UvRg?e=6BUtzg).

[Large Model with 16M Images and 21M Image-Text Pairs](https://1drv.ms/f/s!Aj0m85zoxqZSgc4Inpb0O5-MG-pRDw?e=LqGtFN).

# Text Tokenizer
We use a [bpe tokenizer](https://1drv.ms/f/s!Aj0m85zoxqZSgc1wQgFRv8kN6YKcJw?e=HYoeWs) pre-trained on bookcorpus and wiki.

# Requirements
- Install python3 environment
```angular2html
pip3 install -r requirements.txt
```
- We use pytorch v1.12.1 in implementation and fairscale for gradient checkpoint
- Download raw images from corresponding websites
- Use scripts in data_utils to preprocess different datasets (fix data path and how to read image). Similar with data preparation in [ViLT](https://github.com/dandelin/ViLT/blob/master/DATA.md)

# Fine-tuning
Download [fine-tune json files](https://drive.google.com/file/d/1XFz1Vtz7MCBLn4_1QEojhFJ5Iw3eH3X4/view), the same with [XVLM](https://github.com/zengyan-97/X-VLM)

We use <output_dir> and <output_hdfs_dir> to save checkpoints, and they could be the same path.

\<aug> must be in [None, color, rand].

\<vlmo_config> stands for different models
- config_vlmoB_base64k.json for base model
- config_vlmoL_base64k.json for large model.

\<bs> stands for total batch size (batch size per gpu * gpu_nums).

\<lr> stands for learning rate.

\<lr_mult> stands for multiplying factor on learning rate for parameters except backbone.

\<k_test> stands for selected top_k samples for rerank in Retrieval task.

Use --evaluate in scripts to conduct evaluation only.

## COCO Retrieval
```angular2html
python3 run.py --task=itr_coco --dist=all --checkpoint=<checkpoint_dir> --output_dir=<output_dir> --output_hdfs=<output_hdfs_dir> --augmentation=<aug> --bs=<bs> --lr=<lr> --k_test=<top_k> --lr_mult=<lr_mult>
```

augmentation=color, bs=256, k_test=128, lr_mult=10 is set for COCO Retrieval.

lr=3e-5 for base model and lr=5e-5 for large model.

## Flickr Retrieval
```angular2html
python3 run.py --task=itr_flickr --dist=all --checkpoint=<checkpoint_dir> --output_dir=<output_dir> --output_hdfs=<output_hdfs_dir> --augmentation=<aug> --bs=<bs> --lr=<lr> --k_test=<k_test> --lr_mult=<lr_mult> --vlmo_config=<vlmo_config>
```
augmentation=color, bs=128, lr=1e-5, k_test=128, lr_mult=5 for Flickr Retrieval.

## VQA
```angular2html
python3 run.py --task=vqa --dist=all --config=configs/VQA_beit_480_vg.yaml --checkpoint=<checkpoint_dir> --output_dir=<output_dir> --output_hdfs=<output_hdfs_dir> --vlmo_config=<vlmo_config> --bs=<bs> --lr_mult=<lr_mult> --lr=<lr>
```
bs=128, lr_mult=10, lr=3e-5 for VQA.

## NLVR
```angular2html
python3 run.py --task=NLVR_vit_lrtest_5e-5= --dist=all --config=configs/VQA_beit_480_vg.yaml --checkpoint=<checkpoint_dir> --output_dir=<output_dir> --output_hdfs=<output_hdfs_dir> --beit --augmentation=<aug> --vlmo_config=<vlmo_config> --bs=<bs> --lr_mult=<lr_mult> --lr=<lr> --biattn
```
aug=rand, bs=128, lr=3.5e-5, lr_mult=15 for NLVR.

Use --biattn to enbale bi-attention module.

# Citation
If you find this repository useful, please considering giving ⭐ or citing:
```
@article{eve,
  author    = {Junyi Chen and Longteng Guo and Jia Sun and Shuai Shao and Zehuan Yuan and Liang Lin and Dongyu Zhang},
  title     = {EVE: Efficient Vision-Language Pre-training with Masked Prediction and Modality-Aware MoE},
  journal   = {arXiv preprint arXiv:2308.11971},
  year      = {2023},
}
```