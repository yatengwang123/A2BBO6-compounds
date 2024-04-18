import numpy as np
import pandas as pd
from functools import reduce
from sklearn import preprocessing
from sklearn.metrics import r2_score
from xgboost.sklearn import XGBRegressor
import json

df = pd.read_csv('train_data_set.csv')
df.drop(['A_site_Elements', 'B_site_Elements', "B'_site_Elements", 'O_site_Elements', 'tole3', 'delmu', 'composition',
         'composition_oxid', 'composition', 'chemicalFormula', 'enid', 'A_site', 'B_site', "B'_site", 'O_site'], axis=1,
        inplace=True)
df.drop([], axis=1, inplace=True)
label2 = preprocessing.LabelEncoder()
l1 = pd.concat([df['HOMO_character'], df['LUMO_character']], axis=0)
l1.drop_duplicates(inplace=True)
label2.fit(l1)
df['HOMO_character'] = label2.transform(df['HOMO_character'])
df['LUMO_character'] = label2.transform(df['LUMO_character'])
label1 = preprocessing.LabelEncoder()
l2 = pd.concat([df['HOMO_element'], df['LUMO_element']], axis=0)
l2.drop_duplicates(inplace=True)
label1.fit(l2)
df['HOMO_element'] = label1.transform(df['HOMO_element'])
df['LUMO_element'] = label1.transform(df['LUMO_element'])
X = df.drop('formation_energy', axis=1)
y = df['formation_energy']


def feature_importance_dict(model, df, name):
    feature_importance = {}
    feature_importances = []
    feature_importances_result = {}
    feature_importances_result_dict = {}
    for i in range(20):
        test = df.sample(frac=0.2)
        train = df.drop(test.index, axis=0)
        test_x = test.drop('formation_energy', axis=1)
        test_y = test['formation_energy']
        train_x = train.drop('formation_energy', axis=1)
        train_y = train['formation_energy']
        model.fit(train_x, train_y)
        predict_y = model.predict(test_x)
        for index in model.feature_importances_.argsort():
            feature_importance[train_x.columns[index]] = model.feature_importances_[index]
        feature_importances.append(feature_importance)
    keys = feature_importances[0].keys()
    for key in keys:
        result = 0
        for data in feature_importances:
            result += data[key]
        feature_importances_result[key] = result / len(feature_importances)
    index = np.array(list(feature_importances_result.values())).argsort()[::-1]
    feature = np.array(list(feature_importances_result.keys()))[index]
    value = np.array(list(feature_importances_result.values()))[index]
    if len(feature) == len(value):
        for i in range(len(feature)):
            feature_importances_result_dict[feature[i]] = value[i]
        with open(f'{name}_feature_importance.json', 'w') as f:
            json.dump(feature_importances_result_dict, f)
    return feature, value


def random_choice_train(df, model, number):
    score2 = []
    begin_feature_importance = []
    index = 0 - number
    print(index)
    for i in range(20):
        test = df.sample(frac=0.2)
        train = df.drop(test.index, axis=0)
        test_x = test.drop('formation_energy', axis=1)
        test_y = test['formation_energy']
        train_x = train.drop('formation_energy', axis=1)
        train_y = train['formation_energy']
        model.fit(train_x, train_y)
        predict_y = model.predict(test_x)
        if r2_score(test_y, predict_y) > 0.9:
            begin_feature_importance.append(np.array(train_x.columns[model.feature_importances_.argsort()[index:]]))
        score2.append(r2_score(test_y, predict_y))
    return score2, begin_feature_importance


def correlation3(df):
    r = df.corr(method="spearman")
    return (r)


def filter_corr_null(my_r2):
    corr_null_name = []
    for name in my_r2.columns:
        if my_r2[name].isnull().all():
            print(name)
            if name not in corr_null_name:
                corr_null_name.append(name)
    return corr_null_name


def sperman_rank(sr):
    index_list = []
    for index_col in sr.columns:
        for index_row in sr.index:
            if sr[index_col][index_row] > 0.8:
                if index_col != index_row:
                    if (index_col, index_row, sr[index_col][index_row]) not in index_list and (
                            index_row, index_col, sr[index_col][index_row]) not in index_list:
                        index_list.append((index_col, index_row, sr[index_col][index_row]))
    return index_list


def drop_corr_formation(drop_f2):
    drop_f2_ = []
    for item in drop_f2:
        corr0 = df[[item[0], 'formation_energy']].corr(method="spearman")
        corr1 = df[[item[1], 'formation_energy']].corr(method="spearman")
        if corr0[item[0]]['formation_energy'] > corr1[item[1]]['formation_energy']:
            drop_f2_.append(item[0])
        else:
            drop_f2_.append(item[1])
    return set(drop_f2_)


# XGB_gain特征重要性取并集
model_xgb_gain = XGBRegressor(importance_type='gain')
score_xgb_gain, begin_feature_importance_xgb_gain = random_choice_train(df, model_xgb_gain, 20)
feature_index_any_xgb_gain = reduce(np.union1d, begin_feature_importance_xgb_gain)
feature_index_any_xgb_gain = np.append(feature_index_any_xgb_gain, 'formation_energy')
t2_xgb_gain = df[feature_index_any_xgb_gain]
my_r2_xgb_gain = correlation3(t2_xgb_gain)
corr_null_name_xgb_gain = filter_corr_null(my_r2_xgb_gain)
t2_xgb_gain.drop(corr_null_name_xgb_gain, axis=1, inplace=True)
my_r2_xgb_gain.drop(corr_null_name_xgb_gain, axis=1, inplace=True)
drop_f2_xgb_gain = sperman_rank(my_r2_xgb_gain)
drop_f22_xgb_gain = drop_corr_formation(drop_f2_xgb_gain)
drop_f23_xgb_gain = np.array(list(drop_f22_xgb_gain))
t2_xgb_gain.drop(drop_f23_xgb_gain, axis=1, inplace=True)
my_r21_xgb_gain = correlation3(t2_xgb_gain)
t2_xgb_gain.to_excel('XGB_gain_sperman_data.xlsx', index=False)
feature_gain, value_gain = feature_importance_dict(model=model_xgb_gain, df=t2_xgb_gain,
                                                   name='XGB_gain_feature_importance')

# XGB_weight特征重要性取并集
model_xgb_weight = XGBRegressor(importance_type='weight')
score_xgb_weight, begin_feature_importance_xgb_weight = random_choice_train(df, model_xgb_weight, 30)
feature_index_any_xgb_weight = reduce(np.union1d, begin_feature_importance_xgb_weight)
feature_index_any_xgb_weight = np.append(feature_index_any_xgb_weight, 'formation_energy')
t2_xgb_weight = df[feature_index_any_xgb_weight]
my_r2_xgb_weight = correlation3(t2_xgb_weight)
corr_null_name_xgb_weight = filter_corr_null(my_r2_xgb_weight)
t2_xgb_weight.drop(corr_null_name_xgb_weight, axis=1, inplace=True)
my_r2_xgb_weight.drop(corr_null_name_xgb_weight, axis=1, inplace=True)
drop_f2_xgb_weight = sperman_rank(my_r2_xgb_weight)
drop_f22_xgb_weight = drop_corr_formation(drop_f2_xgb_weight)
drop_f23_xgb_weight = np.array(list(drop_f22_xgb_weight))
t2_xgb_weight.drop(drop_f23_xgb_weight, axis=1, inplace=True)
my_r21_xgb_weight = correlation3(t2_xgb_weight)
feature_weight, value_weight = feature_importance_dict(model=model_xgb_weight, df=t2_xgb_weight, name='XGB_weight_feature_importance')
t2_xgb_weight.to_excel('XGB_weight_sperman_data.xlsx', index=False)

# cover
model_xgb_cover = XGBRegressor(importance_type='cover')
score_xgb_cover, begin_feature_importance_xgb_cover = random_choice_train(df, model_xgb_cover, 30)
feature_index_any_xgb_cover = reduce(np.union1d, begin_feature_importance_xgb_cover)
feature_index_any_xgb_cover = np.append(feature_index_any_xgb_cover, 'formation_energy')
t2_xgb_cover = df[feature_index_any_xgb_cover]
my_r2_xgb_cover = correlation3(t2_xgb_cover)
corr_null_name_xgb_cover = filter_corr_null(my_r2_xgb_cover)
t2_xgb_cover.drop(corr_null_name_xgb_cover, axis=1, inplace=True)
my_r2_xgb_cover.drop(corr_null_name_xgb_cover, axis=1, inplace=True)
drop_f2_xgb_cover = sperman_rank(my_r2_xgb_cover)
drop_f22_xgb_cover = drop_corr_formation(drop_f2_xgb_cover)
drop_f23_xgb_cover = np.array(list(drop_f22_xgb_cover))
t2_xgb_cover.drop(drop_f23_xgb_cover, axis=1, inplace=True)
my_r21_xgb_cover = correlation3(t2_xgb_cover)
feature_cover, value_cover = feature_importance_dict(model=model_xgb_cover, df=t2_xgb_cover, name='XGB_cover_feature_importance')
t2_xgb_cover.to_excel('XGB_cover_sperman_data.xlsx', index=False)
