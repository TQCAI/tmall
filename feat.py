#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import os

import pandas as pd
from joblib import load, dump

from fesys import FeaturesBuilder

feat_builder: FeaturesBuilder = load('feat_builder.pkl')
train_df = pd.read_csv('data_format1/train_format1.csv')
user_info = pd.read_pickle('user_info.pkl')
train_df = train_df.merge(user_info)
origin_map = feat_builder.pk2df
used_keys = [('user_id',), ('merchant_id',), ('user_id', 'merchant_id'), ('age_range',), ('gender',)]
feat_builder.pk2df = {k: feat_builder.pk2df[k] for k in used_keys}
# 商家下面品牌与类别的比例
feat_builder2_fname = 'feat_builder2.pkl'
if os.path.exists(feat_builder2_fname):
    feat_builder2 = load(feat_builder2_fname)
else:
    feat_builder2 = FeaturesBuilder(pd.read_pickle('user_log.pkl'))
    for pk in ['merchant_id']:  # 商家5k个，用户10w个
        feat_builder2.buildCountFeatures(pk, ['brand_id', 'cat_id'])
    feat_builder2.reduce_mem_usage()
    dump(feat_builder2, feat_builder2_fname)
# =======================================================================
# 把 cat brand 的特征 做到 merchant 里面
new_feat = pd.DataFrame()
new_feat['merchant_id'] = feat_builder.pk2df[('merchant_id',)]['merchant_id']
origin_map[('cat_id',)]
train_df2 = feat_builder.outputFeatures(train_df)
print(train_df2)
