#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump

from fesys import FeaturesBuilder

warnings.filterwarnings("ignore")

user_log = pd.read_pickle('data/user_log.pkl')

feat_builder = FeaturesBuilder(user_log)
unique_udf = ('unique', lambda x: len(set(x)))
rebuy_udf = (f'rebuy', lambda x: sum([cnt for cnt in Counter(x).values() if cnt > 1]))

indicator2action_type = {
    'click': 0,
    'add_car': 1,
    'purchase': 2,
    'favorite': 3,
}


# merchant_id, gender, age_range
# 最早最晚出现时间
#
def freq_stat_info(seq):
    cnt = Counter(seq)
    size = len(seq)
    freq = [v / size for v in cnt.values()]
    return np.min(freq), np.mean(freq), np.max(freq), np.std(freq)


freq_stat_info_names = ["freq_min", "freq_mean", "freq_max", "freq_std"]

core_ids = ["user_id", "merchant_id"]
item_ids = ["item_id", "cat_id", "brand_id"]

for indicator in ["purchase", None]:
    # where action_type = 'purchase'
    if indicator is not None:
        action_type = indicator2action_type[indicator]
        feat_builder.core_df = user_log[user_log['action_type'] == action_type]
    else:
        feat_builder.core_df = user_log
    # 对label的统计特征
    for pk in core_ids:
        feat_builder.buildCountFeatures(pk, "label", prefix=indicator)

feat_builder.core_df = None
dump(feat_builder, "data/feat_builder3.pkl")
os.system('google-chrome https://ssl.gstatic.com/dictionary/static/sounds/oxford/ok--_gb_1.mp3')

# 统计双11 3天时间内的一些重要特征 (记录数占24%)
d11_user_log = user_log.query('time_stamp_int in (1110, 1111, 1112)')
feat_builder.core_df = d11_user_log
core_df = d11_user_log
for indicator in ["purchase", None]:
    # where action_type = 'purchase'
    if indicator is not None:
        action_type = indicator2action_type[indicator]
        feat_builder.core_df = user_log[user_log['action_type'] == action_type]
    else:
        feat_builder.core_df = user_log
    # 改indicator名
    if indicator is None:
        indicator = "d11"
    else:
        indicator = f"d11-{indicator}"
    # 预判前缀
    prefix = f"{indicator}-"
    # 以【用户，商铺，【用户 商铺】】为主键，对【items】等算n unique计数
    for pk in core_ids:
        target = [id_ for id_ in core_ids + item_ids if id_ != pk]
        feat_builder.buildCountFeatures(
            pk, target, dummy=False,
            agg_funcs=[unique_udf],
            prefix=indicator)
    feat_builder.buildCountFeatures(core_ids, item_ids, dummy=False, agg_funcs=[unique_udf], prefix=indicator)
    # 用户，商铺，用户商铺的行为比例
    if indicator == "d11":
        for pk in core_ids + [core_ids]:
            feat_builder.buildCountFeatures(pk, 'action_type', prefix=indicator)  # , agg_funcs=[unique_udf] 感觉没必要
    # 双11期间的复购 (不计算多重复购的统计信息了)
    if indicator != "d11":
        feat_builder.buildCountFeatures('user_id', 'merchant_id', dummy=False,
                                        agg_funcs=[rebuy_udf], prefix=indicator)
        feat_builder.buildCountFeatures('merchant_id', 'user_id', dummy=False,
                                        agg_funcs=[rebuy_udf], prefix=indicator)
        feat_builder.addOperateFeatures(f'{prefix}user_rebuy_ratio',
                                        f"lambda x: x['{prefix}user_id-merchant_id-rebuy'] / x['{prefix}user_id-cnt']")
        feat_builder.addOperateFeatures(f'{prefix}merchant_rebuy_ratio',
                                        f"lambda x: x['{prefix}merchant_id-user_id-rebuy'] / x['{prefix}merchant_id-cnt']")
    # 双11期间的用户、商铺比例特征
    feat_builder.addOperateFeatures(f'{prefix}users_div_merchants',
                                    f"lambda x: x['{prefix}user_id-cnt'] / x['{prefix}merchant_id-cnt']")
    feat_builder.addOperateFeatures(f'{prefix}merchants_div_users',
                                    f"lambda x: x['{prefix}user_id-cnt'] / x['{prefix}merchant_id-cnt']")
feat_builder.reduce_mem_usage()
del feat_builder.core_df
dump(feat_builder, "data/feat_builder4.pkl")
print("总特征数：", feat_builder.n_features)
os.system('google-chrome https://ssl.gstatic.com/dictionary/static/sounds/oxford/ok--_gb_1.mp3')
