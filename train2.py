#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import os

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from joblib import load
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from fesys import FeaturesBuilder


def calc_uv_cosine_mv(df):
    df['uv_cosine_mv'] = np.sum(df[merchant_w2v_col].values * df[user_w2v_col].values, axis=1) / \
                         (np.linalg.norm(df[merchant_w2v_col].values, axis=1) *
                          np.linalg.norm(df[user_w2v_col].values, axis=1))


# def calc_vec_sub(df):
#     df = df.merge(user_w2v_n, 'left', on='user_id')
#     df = df.merge(merchant_w2v_n, 'left', on='merchant_id')
#     df[sub_col] = df[user_w2v_n_col].values - df[merchant_w2v_n_col].values
#     df.drop(user_w2v_n_col + merchant_w2v_n_col, axis=1, inplace=True)
#     return df


feat_builder: FeaturesBuilder = load('data/feat_builder.pkl')
train_df = pd.read_csv('data_format1/train_format1.csv')
user_info = pd.read_pickle('data/user_info.pkl')
train_df = train_df.merge(user_info, 'left', on='user_id')
origin_map = feat_builder.pk2df
used_keys = [('user_id',), ('merchant_id',), ('user_id', 'merchant_id'), ('age_range',), ('gender',)]
feat_builder.pk2df = {k: feat_builder.pk2df[k] for k in used_keys}
# 商家下面品牌与类别的比例
# feat_builder2_fname = 'feat_builder2.pkl'
# if os.path.exists(feat_builder2_fname):
#     feat_builder2 = load(feat_builder2_fname)
# else:
#     feat_builder2 = FeaturesBuilder(pd.read_pickle('data/user_log.pkl'))
#     for pk in ['merchant_id']:  # 商家5k个，用户10w个
#         feat_builder2.buildCountFeatures(pk, ['brand_id', 'cat_id'])
#     feat_builder2.reduce_mem_usage()
#     dump(feat_builder2, feat_builder2_fname)
# =======================================================================
# 把 cat brand 的特征 做到 merchant 里面
new_feat = pd.DataFrame()
new_feat['merchant_id'] = feat_builder.pk2df[('merchant_id',)]['merchant_id']
# origin_map[('cat_id',)]
train = feat_builder.outputFeatures(train_df)
merchant_w2v = pd.read_pickle('data/merchant_n2v.pkl')
user_w2v = pd.read_pickle('data/user_n2v.pkl')
merchant_w2v_col = merchant_w2v.columns.tolist()[1:]
user_w2v_col = user_w2v.columns.tolist()[1:]
# 加载降维后的特征
# merchant_w2v_n = pd.read_pickle('data/merchant_w2v_dec_10.pkl')
# user_w2v_n = pd.read_pickle('data/user_w2v_dec_10.pkl')
# merchant_w2v_n_col = merchant_w2v_n.columns.tolist()[1:]
# user_w2v_n_col = user_w2v_n.columns.tolist()[1:]
# sub_col = [f'{c1}-sub-{c2}' for c1, c2 in zip(user_w2v_n_col, merchant_w2v_n_col)]
# exit(0)
# 改格式用来和w2v表拼接
y = train.pop('label')
boruta = load('data/boruta.pkl')
# 删掉不必要的特征
id_c = ['user_id', 'merchant_id']

ids = train[id_c]
train = boruta.transform(train, return_df=True)
train[id_c] = ids
train = train.merge(user_w2v, 'left', on='user_id')
train = train.merge(merchant_w2v, 'left', on='merchant_id')
# 用户与商家的余弦距离
calc_uv_cosine_mv(train)
# 用户embd-商家embd
# train = calc_vec_sub(train)


gbm = LGBMClassifier(random_state=0)
cv = StratifiedKFold(5, True, 0)
# score = cross_val_score(gbm, train, y, cv=cv, scoring='roc_auc').mean()
# print(score)
# print(score)
bc = BalancedBaggingClassifier(
    gbm, random_state=0,
    oob_score=True,  # warm_start=True
)

prediction = pd.read_csv('data_format1/test_format1.csv')
prediction.pop('prob')
test_df = prediction.merge(user_info, 'left', on='user_id')
test = feat_builder.outputFeatures(test_df)

ids = test[id_c]
test = boruta.transform(test, return_df=True)
test[id_c] = ids
test = test.merge(user_w2v, 'left', on='user_id')
test = test.merge(merchant_w2v, 'left', on='merchant_id')
# 用户与商家的余弦距离
calc_uv_cosine_mv(test)
# 用户embd-商家embd
# test = calc_vec_sub(test)

model = bc.fit(train, y)
y_pred = bc.predict_proba(test)
prediction['prob'] = y_pred[:, 1]
prediction.to_csv('predictions/prediction_sub.csv', index=False)
os.system('google-chrome https://ssl.gstatic.com/dictionary/static/sounds/oxford/ok--_gb_1.mp3')
print(bc)
