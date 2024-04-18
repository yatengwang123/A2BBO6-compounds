import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

df_xgb = pd.read_excel('XGB_gain_sperman_data.xlsx')
df_rf = pd.read_excel('RF_sperman_data.xlsx')
df_gbr = pd.read_excel('GBR_sperman_data.xlsx')
df = pd.read_csv('train_data_set.csv')
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
val = pd.read_csv('val_data_set.csv')
val['HOMO_character'] = label2.transform(val['HOMO_character'])
val['LUMO_character'] = label2.transform(val['LUMO_character'])
val['HOMO_element'] = label1.transform(val['HOMO_element'])
val['LUMO_element'] = label1.transform(val['LUMO_element'])
model_xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.15)
model_rf = RandomForestRegressor(n_estimators=100, criterion='squared_error', min_samples_split=2,
                                 min_samples_leaf=1, max_features=1.0, max_depth=28)
model_gbr = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                      min_samples_split=2, min_samples_leaf=1, max_depth=3)
test2 = val.copy()
train2 = df.copy()
# xgb模型的训练和预测
test = test2[df_xgb.columns]
train = train2[df_xgb.columns]
test_x = test.drop('formation_energy', axis=1)
test_y = test['formation_energy']
train_x = train.drop('formation_energy', axis=1)
train_y = train['formation_energy']
model_xgb.fit(train_x, train_y)
predict_y = model_xgb.predict(test_x)
r2 = r2_score(test_y, predict_y)
mae = mean_absolute_error(test_y, predict_y)
print('R2=', r2)
print('MAE=', mae)
df_comp = pd.DataFrame(test_y)
df_comp.rename(columns={'formation_energy': 'fromation_energy_comp'}, inplace=True)
df_comp.reset_index(inplace=True, drop=True)
df_predict = pd.DataFrame(predict_y)
df_predict.rename(columns={0: 'fromation_energy_predcit'}, inplace=True)
df_predict.reset_index(inplace=True, drop=True)
df = pd.concat([df_comp, df_predict], axis=1)
df.to_csv('xgb_predict_comp.csv')

# rf模型的训练和预测
test = test2[df_xgb.columns]
train = train2[df_xgb.columns]
test_x = test.drop('formation_energy', axis=1)
test_y = test['formation_energy']
train_x = train.drop('formation_energy', axis=1)
train_y = train['formation_energy']
model_rf.fit(train_x, train_y)
predict_y = model_rf.predict(test_x)
r2 = r2_score(test_y, predict_y)
mae = mean_absolute_error(test_y, predict_y)
print('R2=', r2)
print('MAE=', mae)
df_comp = pd.DataFrame(test_y)
df_comp.rename(columns={'formation_energy': 'fromation_energy_comp'}, inplace=True)
df_comp.reset_index(inplace=True, drop=True)
df_predict = pd.DataFrame(predict_y)
df_predict.rename(columns={0: 'fromation_energy_predcit'}, inplace=True)
df_predict.reset_index(inplace=True, drop=True)
df = pd.concat([df_comp, df_predict], axis=1)
df.to_csv('rf_predict_comp.csv')

# GBR模型的训练和预测
test = test2[df_xgb.columns]
train = train2[df_xgb.columns]
test_x = test.drop('formation_energy', axis=1)
test_y = test['formation_energy']
train_x = train.drop('formation_energy', axis=1)
train_y = train['formation_energy']
model_gbr.fit(train_x, train_y)
predict_y = model_gbr.predict(test_x)
r2 = r2_score(test_y, predict_y)
mae = mean_absolute_error(test_y, predict_y)
print('R2=', r2)
print('MAE=', mae)
df_comp = pd.DataFrame(test_y)
df_comp.rename(columns={'formation_energy': 'fromation_energy_comp'}, inplace=True)
df_comp.reset_index(inplace=True, drop=True)
df_predict = pd.DataFrame(predict_y)
df_predict.rename(columns={0: 'fromation_energy_predcit'}, inplace=True)
df_predict.reset_index(inplace=True, drop=True)
df = pd.concat([df_comp, df_predict], axis=1)
df.to_csv('gbr_predict_comp.csv')
