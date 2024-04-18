import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
import json
import joblib
from xgboost.sklearn import XGBRegressor

df = pd.read_csv('train_data_set.csv')
df.drop(['A_site_Elements', 'B_site_Elements', "B'_site_Elements", 'O_site_Elements', 'tole3', 'delmu', 'composition',
         'composition_oxid', 'composition', 'chemicalFormula', 'enid', 'A_site', 'B_site', "B'_site", 'O_site'], axis=1,
        inplace=True)
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

min_features_to_select = 1

# RandomForestRegressor Feature Number
model_rf = RandomForestRegressor()
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

rfv_rf = RFECV(estimator=model_rf,
               step=1,
               cv=KFold(5, shuffle=True),
               scoring='r2',
               min_features_to_select=min_features_to_select,
               importance_getter='feature_importances_')
selector_rf = rfv_rf.fit(X, y)
joblib.dump(selector_rf, 'selector_rf.model')
cv_result = {}
for item in selector_rf.cv_results_:
    cv_result[item] = list(rfv_rf.cv_results_[item])
with open('rfv_rf2.json', 'w') as f:
    json.dump(cv_result, f)

# GradientBoostingRegressor Feature Number
model_gbr = GradientBoostingRegressor()
rfv_gbr = RFECV(estimator=model_gbr,
                step=1,
                cv=KFold(5, shuffle=True),
                scoring='r2',
                min_features_to_select=min_features_to_select,
                importance_getter='feature_importances_')
selector_gbr = rfv_gbr.fit(X, y)
cv_result2 = {}
for item in selector_gbr.cv_results_:
    cv_result2[item] = list(selector_gbr.cv_results_[item])
with open('rfv_gbr.json', 'w') as f:
    json.dump(cv_result2, f)
joblib.dump(selector_gbr, 'selector_gbr.model')

# XGB feature number
model_xgb_gain = XGBRegressor(importance_type='gain', tree_method='gpu_hist')
model_xgb_weight = XGBRegressor(importance_type='weight', tree_method='gpu_hist')
model_xgb_cover = XGBRegressor(importance_type='cover', tree_method='gpu_hist')

# gain
rfv_xgb_gain = RFECV(estimator=model_xgb_gain,
                     step=1,
                     cv=KFold(5, shuffle=True),
                     scoring='r2',
                     min_features_to_select=min_features_to_select,
                     importance_getter='feature_importances_', n_jobs=-1)
selector_xgb_gain = rfv_xgb_gain.fit(X, y)

cv_result_gain = {}
for item in selector_xgb_gain.cv_results_:
    cv_result_gain[item] = list(selector_xgb_gain.cv_results_[item])
with open('rfv_xgb_gain.json', 'w') as f:
    json.dump(cv_result_gain, f)
joblib.dump(selector_xgb_gain, 'selector_xgb_gain.model')

# weight
rfv_xgb_weight = RFECV(estimator=model_xgb_weight,
                       step=1,
                       cv=KFold(5, shuffle=True),
                       scoring='r2',
                       min_features_to_select=min_features_to_select,
                       importance_getter='feature_importances_', n_jobs=-1)
selector_xgb_weight = rfv_xgb_weight.fit(X, y)

cv_result_weight = {}
for item in selector_xgb_weight.cv_results_:
    cv_result_weight[item] = list(selector_xgb_weight.cv_results_[item])
with open('rfv_xgb_weight.json', 'w') as f:
    json.dump(cv_result_weight, f)
joblib.dump(selector_xgb_weight, 'selector_xgb_weight.model')

# cover
rfv_xgb_cover = RFECV(estimator=model_xgb_cover,
                      step=1,
                      cv=KFold(5, shuffle=True),
                      scoring='r2',
                      min_features_to_select=min_features_to_select,
                      importance_getter='feature_importances_')
selector_xgb_cover = rfv_xgb_cover.fit(X, y)

cv_result_cover = {}
for item in selector_xgb_cover.cv_results_:
    cv_result_cover[item] = list(selector_xgb_cover.cv_results_[item])
with open('rfv_xgb_cover.json', 'w') as f:
    json.dump(cv_result_cover, f)
joblib.dump(selector_xgb_cover, 'selector_xgb_cover.model')
