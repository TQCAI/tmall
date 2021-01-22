#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import pandas as pd

user_log = pd.read_pickle("user_log.pkl")
user_log.sort_values('time_stamp', inplace=True)
func = lambda x: " ".join(map(str, x))
merchant_seq = user_log.groupby("user_id").agg({"merchant_id": func}). \
    rename(columns={"merchant_id": "merchant_seq"}).reset_index()
user_seq = user_log.groupby("merchant_id").agg({"user_id": func}). \
    rename(columns={"user_id": "user_seq"}).reset_index()
merchant_seq.to_pickle("merchant_seq.pkl")
user_seq.to_pickle("user_seq.pkl")
