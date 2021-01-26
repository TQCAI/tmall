#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-25
# @Contact    : qichun.tang@bupt.edu.cn
import os
from functools import partial

import gensim
import numpy as np
import pandas as pd
from joblib import load, dump

pk2vcr = {}
core_ids = ['user_id', 'merchant_id']
item_ids = ['brand_id', 'item_id', 'cat_id']

n_user = 424170
n_merchant = 4995


def weighted_sum_vectors(x: dict, model: gensim.models.Word2Vec):
    # 已经考虑权重了，需要mean吗？
    return np.array([model.wv.get_vector(str(k)) * w for k, w in x.items()]). \
        mean(axis=0)  # 如果写成mean，如果用户浏览了大量的物品，向量的模就会变小
    #   ↑ mean 还是 sum


# 直接造相似度出来？
for pk in core_ids:
    pk2vcr[pk] = pd.read_pickle(f'data/{pk}_value_counts_ratio.pkl')
k2model = load(f'data/k2model.pkl')
for id_ in item_ids:
    print(id_)
    model = k2model[id_]
    for pk in core_ids:
        print(pk)
        df = pk2vcr[pk]
        df[f"{id_}_vectors"] = df[id_].apply(partial(weighted_sum_vectors, model=model))
        df.pop(id_)
        pk2vcr[pk] = df

dump(pk2vcr, "data/pk2vcr2.pkl")
os.system('google-chrome https://ssl.gstatic.com/dictionary/static/sounds/oxford/ok--_gb_1.mp3')
