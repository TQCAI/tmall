#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-23
# @Contact    : qichun.tang@bupt.edu.cn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

train = pd.read_csv('train_all.csv')
test = pd.read_csv('test_all.csv')
y = train.pop('label')

gbm = LGBMClassifier(random_state=0)
cv = StratifiedKFold(5, True, 0)
score = cross_val_score(gbm, train, y, cv=cv, scoring='roc_auc').mean()
print(score)