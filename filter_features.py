#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-24
# @Contact    : qichun.tang@bupt.edu.cn
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import ExtraTreesClassifier

boruta = BorutaPy(
    ExtraTreesClassifier(max_depth=5, n_jobs=4),
    n_estimators='auto', max_iter=20, random_state=0, verbose=1)

train = pd.read_pickle('data/train.pkl')
y = train.pop('label')
boruta.fit(train, y)
print(boruta)
