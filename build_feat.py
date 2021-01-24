#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import warnings
from collections import Counter

import pandas as pd
from joblib import dump

from fesys import FeaturesBuilder

warnings.filterwarnings("ignore")

user_log = pd.read_pickle('user_log.pkl')

feat_builder = FeaturesBuilder(user_log)
unique_udf = ('unique', lambda x: len(set(x)))
timediff_udf = ('timediff', lambda x: (max(x) - min(x)).days)
mostlike_udf = ('mostlike', lambda x: Counter(x).most_common(1)[0][0] if len(x) else 0)
max_rebuy_times = 4
rebuy_ranges = list(range(1, 11, 1))
rebuy_udf_map = {
    times: (f'rebuy{times}', lambda x: sum([cnt for cnt in Counter(x).values() if cnt > times]))
    for times in rebuy_ranges
}
item_feats = ['merchant_id', 'item_id', 'brand_id', 'cat_id']
sub_item_feats = ['item_id', 'brand_id', 'cat_id']
user_feats = ['age_range', 'gender']
cross_feats = [['user_id', 'merchant_id']]
indicator2action_type = {
    'click': 0,
    'add_car': 1,
    'purchase': 2,
    'favorite': 3,
}
for indicator in ["purchase", "add_car", "favorite", None]:
    if indicator is not None:
        action_type = indicator2action_type[indicator]
        feat_builder.core_df = user_log[user_log['action_type'] == action_type]
    else:
        feat_builder.core_df = user_log
    # ==============================================
    # 计算用户和商铺的复购次数（复购率用算子算） # 叫算子吗，不确定
    if indicator is not None:
        for rebuy_times in rebuy_ranges:
            rebuy_udf = (
                f'rebuy{rebuy_times}', lambda x: sum([cnt for cnt in Counter(x).values() if cnt > rebuy_times]))
            feat_builder.buildCountFeatures('user_id', 'merchant_id', dummy=False,
                                            agg_funcs=[rebuy_udf], prefix=indicator)
            feat_builder.buildCountFeatures('merchant_id', 'user_id', dummy=False,
                                            agg_funcs=[rebuy_udf], prefix=indicator)
    # =============================================
    # 【商家】与用户的【年龄，性别】两个特征的交互
    for pk in item_feats:
        feat_builder.buildCountFeatures(pk, user_feats, prefix=indicator,
                                        agg_funcs=['mean', 'max', 'min', 'median', 'std', 'var', unique_udf])
    # 【商家，商品，品牌，类别】与多少【用户】交互过
    for pk in item_feats:
        feat_builder.buildCountFeatures(pk, 'user_id', dummy=False, agg_funcs=[unique_udf], prefix=indicator)
    # =============================================
    # 【用户】与多少【商家，商品，品牌，类别】交互过（去重）
    feat_builder.buildCountFeatures('user_id', item_feats, dummy=False, agg_funcs=[unique_udf], prefix=indicator)
    # 【商家】,【用户，商品】与多少【商品，品牌，类别】交互过（去重）
    for pk in ['merchant_id'] + cross_feats:
        feat_builder.buildCountFeatures('merchant_id', sub_item_feats, dummy=False, agg_funcs=[unique_udf],
                                        prefix=indicator)
    # =============================================
    if indicator is None:
        # 【用户，商家，商品，品牌，类别, 。。。】的【action_type】统计
        for pk in ['user_id'] + item_feats + user_feats + cross_feats:
            feat_builder.buildCountFeatures(pk, 'action_type', agg_funcs=[unique_udf], prefix=indicator)
    # =============================================
    # 【用户，商家，【用户，商家】】每个【月，星期】的互动次数,  持续时间跨度
    for pk in ['user_id', 'merchant_id'] + cross_feats:
        feat_builder.buildCountFeatures(pk, ['month', 'weekday'], agg_funcs=[unique_udf], prefix=indicator)
        feat_builder.buildCountFeatures(pk, ['time_stamp'], dummy=False, agg_funcs=[timediff_udf], prefix=indicator)
    # =============================================
    # 最喜欢特征
    all_features = ['user_id'] + ['month', 'weekday'] + item_feats
    if indicator is None:
        all_features.append('action_type')
    for feat_a in all_features:
        targets = [feat_b for feat_b in all_features if feat_b != feat_a]
        feat_builder.buildCountFeatures(feat_a, targets, dummy=False, agg_funcs=[mostlike_udf], prefix=indicator)
    prefix = ""
    if indicator is not None:
        prefix = f"{indicator}-"
    # 用户在商铺的出现比例, 以及相反
    feat_builder.addOperateFeatures(f'{prefix}users_div_merchants',
                                    f"lambda x: x['{prefix}user_id-cnt'] / x['{prefix}merchant_id-cnt']")
    feat_builder.addOperateFeatures(f'{prefix}merchants_div_users',
                                    f"lambda x: x['{prefix}user_id-cnt'] / x['{prefix}merchant_id-cnt']")
    # 用户和商铺的复购率
    if indicator:
        for rebuy_times in rebuy_ranges:
            feat_builder.addOperateFeatures(f'{prefix}user_rebuy{rebuy_times}_ratio',
                                            f"lambda x: x['{prefix}user_id-merchant_id-rebuy{rebuy_times}'] / x['{prefix}user_id-cnt']")
            feat_builder.addOperateFeatures(f'{prefix}merchant_rebuy{rebuy_times}_ratio',
                                            f"lambda x: x['{prefix}merchant_id-user_id-rebuy{rebuy_times}'] / x['{prefix}merchant_id-cnt']")
    print('finish', indicator)

feat_builder.reduce_mem_usage()
del feat_builder.core_df
dump(feat_builder, "data/feat_builder.pkl")
print("总特征数：", feat_builder.n_features)
