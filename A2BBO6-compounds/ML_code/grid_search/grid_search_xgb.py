import joblib
from sklearn import preprocessing
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import pickle

df = pd.read_csv('train_data_set.csv')
df.drop(['composition', 'composition_oxid', 'composition', 'chemicalFormula', 'enid', 'A_site', 'B_site', "B'_site",
         'O_site'], axis=1, inplace=True)
df.drop(['A_site_Elements', 'B_site_Elements', "B'_site_Elements", 'O_site_Elements'], axis=1, inplace=True)
df.drop(['tole3', 'delmu'], axis=1, inplace=True)
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
train_X = df.drop('formation_energy', axis=1)
train_Y = df['formation_energy']
model = XGBRegressor()
"""
n_estimators=100, max_depth=5, learning_rate=0.15
"""
params = [
    {'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130, 150],
     'max_depth': [2, 3, 4, 5, 6, 7, 8],
     'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.3, 0.35]}
]
grid_search = GridSearchCV(model, param_grid=params, cv=KFold(10, shuffle=True), scoring='r2', verbose=1)
grid_search.fit(train_X, train_Y)
with open('XGB_program.txt', 'a', encoding='utf-8') as f:
    f.write(str(grid_search.best_estimator_._program))
with open('XGB_score_r2.txt', 'a', encoding='utf-8') as f:
    f.write(str(grid_search.best_score_))

with open('XGB_grid_search.pickle', 'wb') as f:
    pickle.dump(grid_search, f)

joblib.dump(grid_search, 'GBR_grid_search.pkl')
with open('XGB_result.txt', 'a', encoding='utf-8') as f:
    for c, v in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        f.write(str(c) + '\n' + str(v))
