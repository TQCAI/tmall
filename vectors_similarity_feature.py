#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-25
# @Contact    : qichun.tang@bupt.edu.cn
import pandas as pd
pk2vcr={}
core_ids = ['user_id', 'merchant_id']
# 直接造相似度出来？
for pk in core_ids:
    pk2vcr[pk]=pd.read_pickle(f'data/{pk}_value_counts_ratio.pkl')

