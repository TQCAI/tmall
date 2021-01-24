#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-24
# @Contact    : qichun.tang@bupt.edu.cn
import pandas as pd

from sklearn.decomposition import PCA

merchant_w2v = pd.read_pickle('data/merchant_n2v.pkl')
user_w2v = pd.read_pickle('data/user_n2v.pkl')

for n_components in [10, 20, 30]:
    # merchant
    merchant_w2v_d = PCA(n_components=n_components).fit_transform(merchant_w2v.values[:, 1:])
    merchant_w2v_n = pd.DataFrame()
    merchant_w2v_n['merchant_id'] = merchant_w2v['merchant_id']
    merchant_w2v_n[[f"merchant_n2v_dec_{i}" for i in range(n_components)]] = merchant_w2v_d
    merchant_w2v_n.to_pickle(f'data/merchant_w2v_dec_{n_components}')
    # user
    user_w2v_d = PCA(n_components=n_components).fit_transform(user_w2v.values[:, 1:])
    user_w2v_n = pd.DataFrame()
    user_w2v_n['user_id'] = user_w2v['user_id']
    user_w2v_n[[f"user_n2v_dec_{i}" for i in range(n_components)]] = user_w2v_d
    user_w2v_n.to_pickle(f'data/user_w2v_dec_{n_components}')
