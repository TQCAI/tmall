#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-20
# @Contact    : qichun.tang@bupt.edu.cn
import copy
import warnings
from collections import Counter

import gc
import gensim
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")


def read_csv(file_name, num_rows):
    return pd.read_csv(file_name, nrows=num_rows)


# reduce memory
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


num_rows = None
# num_rows = 200 * 10000  # 1000条测试代码使用
# num_rows = 1000


# 读入数据，内存压缩
train_file = './data_format1/train_format1.csv'
test_file = './data_format1/test_format1.csv'

user_info_file = './data_format1/user_info_format1.csv'
user_log_file = './data_format1/user_log_format1.csv'

train_data = reduce_mem_usage(read_csv(train_file, num_rows))
test_data = reduce_mem_usage(read_csv(test_file, num_rows))

user_info = reduce_mem_usage(read_csv(user_info_file, num_rows))
user_log = reduce_mem_usage(read_csv(user_log_file, num_rows))

# 数据处理

## 合并用户信息

del test_data['prob']
all_data = train_data.append(test_data)
all_data = all_data.merge(user_info, on=['user_id'], how='left')
del train_data, test_data, user_info
gc.collect()

## 用户行为日志信息按时间进行排序

user_log = user_log.sort_values(['user_id', 'time_stamp'])

## 对每个用户的逐个合并所有的item_id, cat_id,seller_id,brand_id,time_stamp, action_type字段

list_join_func = lambda x: " ".join([str(i) for i in x])

agg_dict = {  # group by user_id
    'item_id': list_join_func,
    'cat_id': list_join_func,
    'seller_id': list_join_func,
    'brand_id': list_join_func,
    'time_stamp': list_join_func,
    'action_type': list_join_func
}

rename_dict = {
    'item_id': 'item_path',
    'cat_id': 'cat_path',
    'seller_id': 'seller_path',
    'brand_id': 'brand_path',
    'time_stamp': 'time_stamp_path',
    'action_type': 'action_type_path'
}
# 主键是user_id， 表示用户的浏览记录
user_log_path = user_log.groupby('user_id').agg(agg_dict).reset_index().rename(columns=rename_dict)

all_data_path = all_data.merge(user_log_path, on='user_id')

## 删除数据并回收内存

del user_log
gc.collect()


# 定义数据统计函数

def cnt_(x):
    try:
        return len(x.split(' '))
    except:
        return -1


def nunique_(x):
    try:
        return len(set(x.split(' ')))
    except:
        return -1


def max_(x):
    try:
        return np.max([int(i) for i in x.split(' ')])
    except:
        return -1


def min_(x):
    try:
        return np.min([int(i) for i in x.split(' ')])
    except:
        return -1


def std_(x):
    try:
        return np.std([float(i) for i in x.split(' ')])
    except:
        return -1


def most_n(x, n):
    try:
        return Counter(x.split(' ')).most_common(n)[n - 1][0]
    except:
        return -1


def most_n_cnt(x, n):
    try:
        return Counter(x.split(' ')).most_common(n)[n - 1][1]
    except:
        return -1


###


def user_cnt(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(cnt_)
    return df_data


def user_nunique(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(nunique_)
    return df_data


def user_max(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(max_)
    return df_data


def user_min(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(min_)
    return df_data


def user_std(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(std_)
    return df_data


def user_most_n(df_data, single_col, name, n=1):
    func = lambda x: most_n(x, n)  # 可以用偏函数简写
    df_data[name] = df_data[single_col].apply(func)
    return df_data


def user_most_n_cnt(df_data, single_col, name, n=1):
    func = lambda x: most_n_cnt(x, n)
    df_data[name] = df_data[single_col].apply(func)
    return df_data


# 提取商铺的基本统计特征

all_data_test = all_data_path #.head(2000)  # 用来测试的,不是测试集
# all_data_test = all_data_path
# 统计用户 点击、浏览、加购、购买行为
# 总次数
all_data_test = user_cnt(all_data_test, 'seller_path', 'user_cnt')  # 写错了?
# 不同店铺个数
all_data_test = user_nunique(all_data_test, 'seller_path', 'seller_nunique')
# 不同品类个数
all_data_test = user_nunique(all_data_test, 'cat_path', 'cat_nunique')
# 不同品牌个数
all_data_test = user_nunique(all_data_test, 'brand_path', 'brand_nunique')
# 不同商品个数
all_data_test = user_nunique(all_data_test, 'item_path', 'item_nunique')
# 活跃天数
all_data_test = user_nunique(all_data_test, 'time_stamp_path', 'time_stamp_nunique')
# 不用行为种数
all_data_test = user_nunique(all_data_test, 'action_type_path', 'action_type_nunique')

# 最晚时间
all_data_test = user_max(all_data_test, 'action_type_path', 'time_stamp_max')
# 最早时间
all_data_test = user_min(all_data_test, 'action_type_path', 'time_stamp_min')
# 活跃天数方差
all_data_test = user_std(all_data_test, 'action_type_path', 'time_stamp_std')
# 最早和最晚相差天数
all_data_test['time_stamp_range'] = all_data_test['time_stamp_max'] - all_data_test['time_stamp_min']

# 用户最喜欢的店铺
all_data_test = user_most_n(all_data_test, 'seller_path', 'seller_most_1', n=1)
# 最喜欢的类目
all_data_test = user_most_n(all_data_test, 'cat_path', 'cat_most_1', n=1)
# 最喜欢的品牌
all_data_test = user_most_n(all_data_test, 'brand_path', 'brand_most_1', n=1)
# 最常见的行为动作
all_data_test = user_most_n(all_data_test, 'action_type_path', 'action_type_1', n=1)

# 用户最喜欢的店铺 行为次数
all_data_test = user_most_n_cnt(all_data_test, 'seller_path', 'seller_most_1_cnt', n=1)
# 最喜欢的类目 行为次数
all_data_test = user_most_n_cnt(all_data_test, 'cat_path', 'cat_most_1_cnt', n=1)
# 最喜欢的品牌 行为次数
all_data_test = user_most_n_cnt(all_data_test, 'brand_path', 'brand_most_1_cnt', n=1)
# 最常见的行为动作 行为次数
all_data_test = user_most_n_cnt(all_data_test, 'action_type_path', 'action_type_1_cnt', n=1)


# 分开统计用户的点击，加购，购买，收藏特征
## 不同行为的业务函数定义

def col_cnt_(df_data, columns_list, action_type):
    try:  # df_data: Series
        data_dict = {}  # 比如, columns_list = ['seller_path']

        col_list = copy.deepcopy(columns_list)
        if action_type != None:
            col_list += ['action_type_path']  # 现在是 ['seller_path', 'action_type_path']

        for col in col_list:
            data_dict[col] = df_data[col].split(' ')

        path_len = len(data_dict[col])  # 用户产生了 path_len 个行为

        data_out = []
        for i_ in range(path_len):
            data_txt = ''
            for col_ in columns_list:
                if data_dict['action_type_path'][i_] == action_type:
                    data_txt += '_' + data_dict[col_][i_]
            data_out.append(data_txt)

        return len(data_out)
    except:
        return -1


def col_nuique_(df_data, columns_list, action_type):
    try:
        data_dict = {}

        col_list = copy.deepcopy(columns_list)
        if action_type != None:
            col_list += ['action_type_path']

        for col in col_list:
            data_dict[col] = df_data[col].split(' ')

        path_len = len(data_dict[col])

        data_out = []
        for i_ in range(path_len):
            data_txt = ''
            for col_ in columns_list:
                if data_dict['action_type_path'][i_] == action_type:
                    data_txt += '_' + data_dict[col_][i_]
            data_out.append(data_txt)

        return len(set(data_out))
    except:
        return -1


def user_col_cnt(df_data, columns_list, action_type, name):
    df_data[name] = df_data.apply(lambda x: col_cnt_(x, columns_list, action_type), axis=1)
    return df_data


def user_col_nunique(df_data, columns_list, action_type, name):
    df_data[name] = df_data.apply(lambda x: col_nuique_(x, columns_list, action_type), axis=1)
    return df_data


# 点击次数
all_data_test = user_col_cnt(all_data_test, ['seller_path'], '0', 'user_cnt_0')
# 加购次数
all_data_test = user_col_cnt(all_data_test, ['seller_path'], '1', 'user_cnt_1')
# 购买次数
all_data_test = user_col_cnt(all_data_test, ['seller_path'], '2', 'user_cnt_2')
# 收藏次数
all_data_test = user_col_cnt(all_data_test, ['seller_path'], '3', 'user_cnt_3')

# 不同店铺个数
all_data_test = user_col_nunique(all_data_test, ['seller_path'], '0', 'seller_nunique_0')

# 点击次数 (其实没啥用)
# all_data_test = user_col_cnt(all_data_test,  ['seller_path', 'item_path'], '0', 'user_cnt_0')

# 不同店铺个数
all_data_test = user_col_nunique(all_data_test, ['seller_path', 'item_path'], '0', 'seller_nunique_0')

# 组合特征

all_data_test = user_col_cnt(all_data_test, ['seller_path', 'item_path'], '0', 'user_cnt_0')

# 不同店铺个数
all_data_test = user_col_nunique(all_data_test, ['seller_path', 'item_path'], '0', 'seller_nunique_0')

# cntVec = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 1), max_features=100)
tfidfVec = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,
                           ngram_range=(1, 1), max_features=100)

# 利用countvector，tfidf提取特征

# columns_list = ['seller_path', 'cat_path', 'brand_path', 'action_type_path', 'item_path', 'time_stamp_path']
columns_list = ['seller_path']
for i, col in enumerate(columns_list):
    all_data_test[col] = all_data_test[col].astype(str)
    tfidfVec.fit(all_data_test[col])
    data_ = tfidfVec.transform(all_data_test[col])
    if i == 0:
        data_cat = data_
    else:
        data_cat = sparse.hstack((data_cat, data_))

## 特征重命名 特征合并

df_tfidf = pd.DataFrame(data_cat.toarray())
df_tfidf.columns = ['tfidf_' + str(i) for i in df_tfidf.columns]
all_data_test = pd.concat([all_data_test, df_tfidf], axis=1)

## embeeding特征


# Train Word2Vec model

model = gensim.models.Word2Vec(all_data_test['seller_path'].apply(lambda x: x.split(' ')), size=100, window=5,
                               min_count=5, workers=4)


# model.save("product2vec.model")
# model = gensim.models.Word2Vec.load("product2vec.model")

def mean_w2v_(x, model, size=100):
    try:
        i = 0
        for word in x.split(' '):
            if word in model.wv.vocab:
                i += 1
                if i == 1:
                    vec = np.zeros(size)
                vec += model.wv[word]
        return vec / i
    except:
        return np.zeros(size)


def get_mean_w2v(df_data, columns, model, size):
    data_array = []
    for index, row in df_data.iterrows():
        w2v = mean_w2v_(row[columns], model, size)
        data_array.append(w2v)
    return pd.DataFrame(data_array)


df_embeeding = get_mean_w2v(all_data_test, 'seller_path', model, 100)
df_embeeding.columns = ['embeeding_' + str(i) for i in df_embeeding.columns]

## embeeding特征和原始特征合并

all_data_test = pd.concat([all_data_test, df_embeeding], axis=1)


# stacking特征

# from sklearn.cross_validation import KFold


## stacking 回归特征

def stacking_reg(clf, train_x, train_y, test_x, clf_name, kf, label_split=None):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf", "ada", "gb", "et", "lr"]:
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'eval_metric': 'rmse',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12
                      }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round, evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(z, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression_l2',
                'metric': 'mse',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.03,
                'tree_method': 'exact',
                'seed': 2017,
                'nthread': 12,
                'silent': True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(te_x, num_iteration=model.best_iteration).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:" % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    return train.reshape(-1, 1), test.reshape(-1, 1)


def rf_reg(x_train, y_train, x_valid, kf, label_split=None):
    randomforest = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs=-1, random_state=2017,
                                         max_features="auto", verbose=1)
    rf_train, rf_test = stacking_reg(randomforest, x_train, y_train, x_valid, "rf", kf, label_split=label_split)
    return rf_train, rf_test, "rf_reg"


def ada_reg(x_train, y_train, x_valid, kf, label_split=None):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid, "ada", kf, label_split=label_split)
    return ada_train, ada_test, "ada_reg"


def gb_reg(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingRegressor(learning_rate=0.04, n_estimators=100, subsample=0.8, random_state=2017,
                                     max_depth=5, verbose=1)
    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid, "gb", kf, label_split=label_split)
    return gbdt_train, gbdt_test, "gb_reg"


def et_reg(x_train, y_train, x_valid, kf, label_split=None):
    extratree = ExtraTreesRegressor(n_estimators=600, max_depth=35, max_features="auto", n_jobs=-1, random_state=2017,
                                    verbose=1)
    et_train, et_test = stacking_reg(extratree, x_train, y_train, x_valid, "et", kf, label_split=label_split)
    return et_train, et_test, "et_reg"


def lr_reg(x_train, y_train, x_valid, kf, label_split=None):
    lr_reg = LinearRegression(n_jobs=-1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, "lr", kf, label_split=label_split)
    return lr_train, lr_test, "lr_reg"


def xgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    xgb_train, xgb_test = stacking_reg(xgboost, x_train, y_train, x_valid, "xgb", kf, label_split=label_split)
    return xgb_train, xgb_test, "xgb_reg"


def lgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    lgb_train, lgb_test = stacking_reg(lightgbm, x_train, y_train, x_valid, "lgb", kf, label_split=label_split)
    return lgb_train, lgb_test, "lgb_reg"


# stacking 分类特征

def stacking_clf(clf, train_x, train_y, test_x, clf_name, kf, label_split=None):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]

        if clf_name in ["rf", "ada", "gb", "et", "lr", "knn", "gnb"]:
            clf.fit(tr_x, tr_y)
            pre = clf.predict_proba(te_x)

            train[test_index] = pre[:, 0].reshape(-1, 1)
            test_pre[i, :] = clf.predict_proba(test_x)[:, 0].reshape(-1, 1)

            cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x)
            params = {'booster': 'gbtree',
                      'objective': 'multi:softprob',
                      'eval_metric': 'mlogloss',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      "num_class": 2
                      }

            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round, evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)
                train[test_index] = pre[:, 0].reshape(-1, 1)
                test_pre[i, :] = model.predict(z, ntree_limit=model.best_ntree_limit)[:, 0].reshape(-1, 1)
                cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                'boosting_type': 'gbdt',
                # 'boosting_type': 'dart',
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.03,
                'tree_method': 'exact',
                'seed': 2017,
                "num_class": 2,
                'silent': True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(te_x, num_iteration=model.best_iteration)
                train[test_index] = pre[:, 0].reshape(-1, 1)
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration)[:, 0].reshape(-1, 1)
                cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:" % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    return train.reshape(-1, 1), test.reshape(-1, 1)


def rf_clf(x_train, y_train, x_valid, kf, label_split=None):
    randomforest = RandomForestClassifier(n_estimators=1200, max_depth=20, n_jobs=-1, random_state=2017,
                                          max_features="auto", verbose=1)
    rf_train, rf_test = stacking_clf(randomforest, x_train, y_train, x_valid, "rf", kf, label_split=label_split)
    return rf_train, rf_test, "rf"


def ada_clf(x_train, y_train, x_valid, kf, label_split=None):
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_clf(adaboost, x_train, y_train, x_valid, "ada", kf, label_split=label_split)
    return ada_train, ada_test, "ada"


def gb_clf(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingClassifier(learning_rate=0.04, n_estimators=100, subsample=0.8, random_state=2017,
                                      max_depth=5, verbose=1)
    gbdt_train, gbdt_test = stacking_clf(gbdt, x_train, y_train, x_valid, "gb", kf, label_split=label_split)
    return gbdt_train, gbdt_test, "gb"


def et_clf(x_train, y_train, x_valid, kf, label_split=None):
    extratree = ExtraTreesClassifier(n_estimators=1200, max_depth=35, max_features="auto", n_jobs=-1, random_state=2017,
                                     verbose=1)
    et_train, et_test = stacking_clf(extratree, x_train, y_train, x_valid, "et", kf, label_split=label_split)
    return et_train, et_test, "et"


def xgb_clf(x_train, y_train, x_valid, kf, label_split=None):
    xgb_train, xgb_test = stacking_clf(xgboost, x_train, y_train, x_valid, "xgb", kf, label_split=label_split)
    return xgb_train, xgb_test, "xgb"


def lgb_clf(x_train, y_train, x_valid, kf, label_split=None):
    xgb_train, xgb_test = stacking_clf(lightgbm, x_train, y_train, x_valid, "lgb", kf, label_split=label_split)
    return xgb_train, xgb_test, "lgb"


def gnb_clf(x_train, y_train, x_valid, kf, label_split=None):
    gnb = GaussianNB()
    gnb_train, gnb_test = stacking_clf(gnb, x_train, y_train, x_valid, "gnb", kf, label_split=label_split)
    return gnb_train, gnb_test, "gnb"


def lr_clf(x_train, y_train, x_valid, kf, label_split=None):
    logisticregression = LogisticRegression(n_jobs=-1, random_state=2017, C=0.1, max_iter=200)
    lr_train, lr_test = stacking_clf(logisticregression, x_train, y_train, x_valid, "lr", kf, label_split=label_split)
    return lr_train, lr_test, "lr"


def knn_clf(x_train, y_train, x_valid, kf, label_split=None):
    kneighbors = KNeighborsClassifier(n_neighbors=200, n_jobs=-1)
    knn_train, knn_test = stacking_clf(kneighbors, x_train, y_train, x_valid, "lr", kf, label_split=label_split)
    return knn_train, knn_test, "knn"


# 获取训练和验证数据(为stacking特征做准备)

features_columns = [c for c in all_data_test.columns if
                    c not in ['label', 'prob', 'seller_path', 'cat_path', 'brand_path', 'action_type_path', 'item_path',
                              'time_stamp_path']]
x_train = all_data_test[~all_data_test['label'].isna()][features_columns].values
y_train = all_data_test[~all_data_test['label'].isna()]['label'].values
x_valid = all_data_test[all_data_test['label'].isna()][features_columns].values


# 处理函数值inf以及nan情况

def get_matrix(data):
    where_are_nan = np.isnan(data)
    where_are_inf = np.isinf(data)
    data[where_are_nan] = 0
    data[where_are_inf] = 0
    return data


x_train = np.float_(get_matrix(np.float_(x_train)))
y_train = np.int_(y_train)
x_valid = x_train

# 导入划分数据函数 设stacking特征为5折

from sklearn.model_selection import KFold

folds = 5
seed = 1
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# 使用lgb和xgb分类模型构造stacking特征


clf_list = [lgb_clf, xgb_clf]
clf_list_col = ['lgb_clf', 'xgb_clf']

# 训练模型，获取stacking特征

clf_list = clf_list
column_list = []
train_data_list = []
test_data_list = []
for clf in clf_list:
    train_data, test_data, clf_name = clf(x_train, y_train, x_valid, kf, label_split=None)
    train_data_list.append(train_data)
    test_data_list.append(test_data)
train_stacking = np.concatenate(train_data_list, axis=1)
test_stacking = np.concatenate(test_data_list, axis=1)

# 原始特征和stacking特征合并

train = pd.DataFrame(np.concatenate([x_train, train_stacking], axis=1))
test = np.concatenate([x_valid, test_stacking], axis=1)

# 特征重命名

df_train_all = pd.DataFrame(train)
df_train_all.columns = features_columns + clf_list_col
df_test_all = pd.DataFrame(test)
df_test_all.columns = features_columns + clf_list_col

# 获取数据ID以及特征标签LABEL

df_train_all['label'] = all_data_test['label']

# 训练数据和测试数据保存

df_train_all.to_csv('train_all.csv', header=True, index=False)
df_test_all.to_csv('test_all.csv', header=True, index=False)
