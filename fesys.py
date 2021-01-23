#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-21
# @Contact    : qichun.tang@bupt.edu.cn
import warnings
from typing import List, Dict, Union, Tuple, Callable

import pandas as pd

from utils import reduce_mem_usage

warnings.filterwarnings("ignore")


class FeaturesBuilder():
    def __init__(self, core_df):
        self.core_df: pd.DataFrame = core_df
        self.pk2df: Dict[str, pd.DataFrame] = {}  # primaryKeyToDataFrame
        self.op_feats: List[Tuple[str, Callable]] = []

    @property
    def n_features(self):
        res = 0
        for pk, df in self.pk2df.items():
            res += df.shape[1] - len(pk)
        res += len(self.op_feats)
        return res

    def reduce_mem_usage(self):
        for pk in self.pk2df:
            self.pk2df[pk] = reduce_mem_usage(self.pk2df[pk])

    def buildCountFeatures(
            self,
            primaryKey: Union[List[str], str],
            countValues: Union[List[str], str, None],
            countPK=True,
            dummy=True,
            ratio=True,
            agg_funcs=None,  # 注意， 如果countValues为离散特征或计数特征请谨慎使用，所以默认为None
            prefix=None
            # descriptions=None # 如果不为空，长度需要与countValues一致
    ):
        if isinstance(primaryKey, str):
            primaryKey = [primaryKey]
        if isinstance(countValues, str):
            countValues = [countValues]
        # 如果不存在主键对应的DF，创建新的
        t_pk = tuple(primaryKey)
        if t_pk not in self.pk2df:
            df_agg = self.core_df[primaryKey].drop_duplicates().sort_values(by=primaryKey). \
                reset_index(drop=True)
            self.pk2df[t_pk] = df_agg
        # 主键列名
        pk_col = "-".join(primaryKey)
        if prefix:
            pk_col = f"{prefix}-{pk_col}"
        # 根据规则对参数进行校验
        if not countValues:
            dummy = False
            agg_funcs = None
        if dummy == False or countPK == False:
            ratio = False
        # 先对主键进行统计
        pk_cnt_col = f"{pk_col}-cnt"
        if countPK and pk_cnt_col not in self.pk2df[t_pk].columns:
            pk_cnt_df = self.core_df.groupby(primaryKey).size().reset_index().rename(columns={0: pk_cnt_col})
            self.pk2df[t_pk] = self.pk2df[t_pk].merge(pk_cnt_df, on=primaryKey)
        # 对聚集函数进行处理
        agg_funcs_ = []
        agg_cols_ = []
        if agg_funcs is not None:
            for agg_func in agg_funcs:
                if isinstance(agg_func, str):
                    agg_funcs_.append(agg_func)
                    agg_cols_.append(agg_func)
                elif isinstance(agg_func, tuple):
                    assert len(agg_func) == 2 and isinstance(agg_func[0], str) and callable(agg_func[1])
                    agg_funcs_.append(agg_func[1])
                    agg_cols_.append(agg_func[0])
                else:
                    raise ValueError
        # 对countValues进行处理
        if not countValues:
            countValues = []
        # 对descriptions进行处理
        # if descriptions is None:
        #     descriptions=['']*len(countValues)
        # for循环，对每个要计算的列进行统计
        for countValue in countValues:
            pk_val_col = f"{pk_col}-{countValue}"
            if agg_funcs_:
                cur_agg_cols_ = [f"{pk_val_col}-{agg_col}" for agg_col in agg_cols_]
                df_agg = self.core_df.groupby(primaryKey).agg({countValue: agg_funcs_}).reset_index()
                df_agg.columns = primaryKey + cur_agg_cols_
                # 将除0产生的nan替换为0
                df_agg.fillna(0, inplace=True)
                self.pk2df[t_pk] = self.pk2df[t_pk].merge(df_agg, on=primaryKey)
            dummy_columns = []
            if dummy:
                # 对values计数，得到dummy特征
                pk_val_cnt_df = self.core_df.groupby(primaryKey + [countValue]).size().reset_index(). \
                    rename(columns={0: pk_val_col})
                pk_cnt_df_dummy = pd.get_dummies(pk_val_cnt_df, columns=[countValue], prefix=pk_val_col)
                dummy_columns = pk_cnt_df_dummy.columns.to_list()[len(primaryKey) + 1:]
                for column in dummy_columns:
                    pk_cnt_df_dummy[column] *= pk_cnt_df_dummy[pk_val_col]
                pk_cnt_df_dummy = pk_cnt_df_dummy.groupby(primaryKey).sum().reset_index().drop(pk_val_col, axis=1)
                self.pk2df[t_pk] = self.pk2df[t_pk].merge(pk_cnt_df_dummy, on=primaryKey)
            if ratio and dummy_columns:
                ratio_columns = [f"{dummy_column}-div-{pk_cnt_col}" for dummy_column in dummy_columns]
                for ratio_column, dummy_column in zip(ratio_columns, dummy_columns):
                    self.pk2df[t_pk][ratio_column] = self.pk2df[t_pk][dummy_column] / self.pk2df[t_pk][pk_cnt_col]
                # 将除0产生的nan替换为0
                self.pk2df[t_pk].fillna(0, inplace=True)

    def addOperateFeatures(
            self,
            new_feature_name: str,
            df_apply_func: Union[Callable, str]
    ):
        # 记得处理nan
        self.op_feats.append([new_feature_name, df_apply_func])

    def outputFeatures(self, base_df: pd.DataFrame, apply_op=True):
        df = base_df
        pk_list = list(self.pk2df.keys())
        pk_list.sort()
        for pk in pk_list:
            df = df.merge(self.pk2df[pk], 'left', on=pk)
        if apply_op:
            self.applyOperateFeatures(df)
        return df

    def applyOperateFeatures(self, base_df: pd.DataFrame):
        for name, func in self.op_feats:
            if isinstance(func, str):
                func = eval(func)
            base_df[name] = func(base_df)
        return base_df
