import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from functools import reduce
from sklearn import preprocessing
from sklearn.metrics import r2_score
import json

df = pd.read_csv('train_data_set.csv')

df.drop(['A_site_Elements', 'B_site_Elements', "B'_site_Elements", 'O_site_Elements', 'composition', 'composition_oxid',
         'composition', 'tole3', 'delmu', 'chemicalFormula', 'enid', 'A_site', 'B_site', "B'_site", 'O_site'], axis=1,
        inplace=True)


def feature_importance_dict(model, df):
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
        with open('GBR_feature_importance.json', 'w') as f:
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

model_gbr = GradientBoostingRegressor()
score_gbr, begin_feature_importance_gbr = random_choice_train(df, model_gbr, 30)
feature_index_any_gbr = reduce(np.union1d, begin_feature_importance_gbr)
feature_index_any_gbr = np.append(feature_index_any_gbr, 'formation_energy')
t2_gbr = df[feature_index_any_gbr]
my_r2_gbr = correlation3(t2_gbr)
corr_null_name_gbr = filter_corr_null(my_r2_gbr)
t2_gbr.drop(corr_null_name_gbr, axis=1, inplace=True)
my_r2_gbr.drop(corr_null_name_gbr, axis=1, inplace=True)
drop_f2_gbr = sperman_rank(my_r2_gbr)
drop_f22_gbr = drop_corr_formation(drop_f2_gbr)
drop_f23_gbr = np.array(list(drop_f22_gbr))
t2_gbr.drop(drop_f23_gbr, axis=1, inplace=True)
my_r21 = correlation3(t2_gbr)
feature, value = feature_importance_dict(model=model_gbr, df=t2_gbr)
t2_gbr.to_excel('GBR_sperman_data.xlsx', index=False)
