#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-24
# @Contact    : qichun.tang@bupt.edu.cn
import pandas as pd
from gensim.models import Word2Vec
from joblib import dump

user_log = pd.read_pickle("data/user_log.pkl")
user_log.sort_values('time_stamp', inplace=True)
print('完成排序')
func = lambda x: " ".join(map(str, x))
# item  : 1 090 390
# cat   : 1 658
# brand : 4 124
k2dim = {  # 根据用户的浏览继续生成物品序列，然后学习对应的词向量
    'brand_id': (30, 5),
    'cat_id': (20, 4),
    'item_id': (100, 10),
}
k2seq = {}
k2model = {}
for k in k2dim:
    k2seq[k] = user_log.groupby("user_id").agg({k: func}).reset_index()
    print(f'完成对{k}序列的数据整理')
dump(k2seq, 'data/k2seq.pkl')
print('开始训练模型')
for k, seq in k2seq.items():
    size, window = k2dim[k]
    model = Word2Vec(
        seq[k].apply(lambda x: x.split(' ')),
        size=size,
        window=window,
        min_count=1,
        workers=12
    )
    k2model[k] = model
    print(f'完成对{k}模型的训练')
dump(k2model, 'data/k2model.pkl')
