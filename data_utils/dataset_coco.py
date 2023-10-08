from utils.hdfs_io import *
import pandas as pd
import json
from PIL import Image
from base64 import b64decode, b64encode
import io
import pyarrow as pa
import gc
from tqdm import tqdm
import os

annos = {'train2014_path':'images/annotations2014/captions_train2014.json', 
    'val2014_path': 'images/annotations2014/captions_val2014.json'} # fix path

num_per_slice = 100000
dataset_root = "save_dir"
filter_list = set()
filter_imgs = json.load(open('finetune/coco_test.json', 'r')) # fix path
for img in filter_imgs:
    filter_list.add(os.path.join('image_path', img['image'])) # fix path
cur_slice = 0
data_list = []
for image_path, anno_path in annos.items():
    print(image_path, 'start')
    slice = image_path.split('/')[-1]
    data = json.load(open(anno_path, 'r'))
    files = {}
    for image in data['images']:
        if image['id'] not in files:
            files[image['id']] = os.path.join(image_path, image['file_name'])
    for anno in tqdm(data['annotations']):
        if files[anno['image_id']] in filter_list:
            print(files[anno['image_id']], 'filter')
            continue
        img = open(files[anno['image_id']], 'rb').read()
        data_list.append((img, anno['caption'], slice+"_"+str(anno["image_id"])))
        if len(data_list) >= num_per_slice:
            dataframe = pd.DataFrame(
                            data_list, columns=["image", "caption", "image_id"],
                        )
            table = pa.Table.from_pandas(dataframe)
            with pa.OSFile(
                    f"{dataset_root}/coco_{cur_slice}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            data_list.clear()
            gc.collect()
            cur_slice += 1
        
if len(data_list):
    dataframe = pd.DataFrame(
                    data_list, columns=["image", "caption", "image_id"],
                )
    table = pa.Table.from_pandas(dataframe)
    with pa.OSFile(
            f"{dataset_root}/coco_{cur_slice}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    del dataframe
    del table
    data_list.clear()
    gc.collect()
    cur_slice += 1