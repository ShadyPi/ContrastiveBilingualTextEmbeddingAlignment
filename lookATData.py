import json

import pandas as pd

with open('./data/WMT18/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
print(data[:5][0])

df = pd.read_json("hf://datasets/mteb/biorxiv-clustering-s2s/test.jsonl.gz", lines=True)
print(len(df))
# total_sum = 0
# for i, row in df.iterrows():
#     total_sum += len(row['sentences'])
# print(total_sum)