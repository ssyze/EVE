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
from collections import defaultdict
from glob import glob
import random

def path2rest(path, iid2captions):
    name = path.split("/")[-1]
    iid = int(name[:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    cdicts = iid2captions[iid]
    captions = [c["phrase"] for c in cdicts]
    widths = [c["width"] for c in cdicts]
    heights = [c["height"] for c in cdicts]
    xs = [c["x"] for c in cdicts]
    ys = [c["y"] for c in cdicts]

    return [
        binary,
        captions,
        widths,
        heights,
        xs,
        ys,
        str(iid),
    ]

captions = json.load(open("images/vg/region_descriptions.json", "r")) # fix path
iid2captions = defaultdict(list)
root = "image_path" # fix path
dataset_root = "save_dir" # fix path
for cap in tqdm(captions):
    cap = cap["regions"]
    for c in cap:
        iid2captions[c["image_id"]].append(c)

paths = list(glob(f"{root}/VG_100K/*.jpg")) + list(
    glob(f"{root}/VG_100K_2/*.jpg"))
random.shuffle(paths)
caption_paths = [
    path for path in paths if int(path.split("/")[-1][:-4]) in iid2captions
]
print(caption_paths[0])

if len(paths) == len(caption_paths):
    print("all images have caption annotations")
else:
    print("not all images have caption annotations")
print(
    len(paths),
    len(caption_paths),
    len(iid2captions),
)

# exit()

bs = [path2rest(path, iid2captions) for path in tqdm(caption_paths)]
dataframe = pd.DataFrame(
    bs,
    columns=["image", "caption", "width", "height", "x", "y", "image_id"],
)
table = pa.Table.from_pandas(dataframe)

os.makedirs(dataset_root, exist_ok=True)
with pa.OSFile(f"{dataset_root}/vg.arrow", "wb") as sink:
    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
        writer.write_table(table)