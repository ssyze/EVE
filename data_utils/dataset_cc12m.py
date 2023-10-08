from utils.hdfs_io import *
import pandas as pd
import json
from PIL import Image
from base64 import b64decode
import io
import pyarrow as pa
import gc
from tqdm import tqdm

def get(paths):
    for filepath in paths:
        with hopen(filepath, 'r') as reader:
            for line in reader:
                yield line.decode()

paths = [
        "data_path"
]
files = hlist_files(paths)
files = [f for f in files if f.find('_SUCCESS') < 0]
print(len(files))
datas = get(files)
data_list = []
num_per_slice = 100000
dataset_root = "save_dir"
cur_slice = 0
for data in tqdm(datas):
    data = json.loads(data)
    binary = b64decode(data['binary']) # Image.open(path) for raw Image
    data_list.append((binary, data['desc'], data['url']))
    if len(data_list) >= num_per_slice:
        dataframe = pd.DataFrame(
                        data_list, columns=["image", "caption", "image_id"],
                    )
        table = pa.Table.from_pandas(dataframe)
        with pa.OSFile(
                f"{dataset_root}/cc12m_{cur_slice}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        data_list.clear()
        gc.collect()
        cur_slice += 1