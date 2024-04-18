import joblib
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import pickle
df=pd.read_csv('train_data_set.csv')
df.drop(['composition','composition_oxid','composition','chemicalFormula','enid','A_site','B_site',"B'_site",'O_site'],axis=1,inplace=True)
df.drop(['A_site_Elements','B_site_Elements',"B'_site_Elements",'O_site_Elements'],axis=1,inplace=True)
df.drop(['tole3','delmu'],axis=1,inplace=True)
label2=preprocessing.LabelEncoder()
l1=pd.concat([df['HOMO_character'],df['LUMO_character']],axis=0)
l1.drop_duplicates(inplace=True)
label2.fit(l1)
df['HOMO_character']=label2.transform(df['HOMO_character'])
df['LUMO_character']=label2.transform(df['LUMO_character'])
label1=preprocessing.LabelEncoder()
l2=pd.concat([df['HOMO_element'],df['LUMO_element']],axis=0)
l2.drop_duplicates(inplace=True)
label1.fit(l2)
df['HOMO_element']=label1.transform(df['HOMO_element'])
df['LUMO_element']=label1.transform(df['LUMO_element'])
train_X=df.drop('formation_energy',axis=1)
train_Y=df['formation_energy']
model=RandomForestRegressor()
"""
n_estimators=100,criterion=’squared_error’, min_samples_split=2,
min_samples_leaf=1, max_features=1.0, max_depth=28"
"""
params=[
    {'n_estimators':[50,60,70,80,90,100,110,120,130,150],
     'max_depth':[25,26,27,28,29,30],
     'min_samples_split':[1,2,4,6,8],
     'min_samples_leaf':[1,2,3,4],}
]
grid_search=GridSearchCV(model,param_grid=params,cv=KFold(5,shuffle=True),scoring='r2',verbose=1)
grid_search.fit(train_X, train_Y)
with open('RF_program.txt', 'a', encoding='utf-8') as f:
    f.write(str(grid_search.best_estimator_._program))
with open('RF_score_r2.txt', 'a', encoding='utf-8') as f:
    f.write(str(grid_search.best_score_))

with open('RF_grid_search.pickle', 'wb') as f:
    pickle.dump(grid_search, f)

joblib.dump(grid_search, 'RF_grid_search.pkl')
with open('RF_result.txt', 'a', encoding='utf-8') as f:
    for c, v in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        f.write(str(c)+'\n'+str(v))