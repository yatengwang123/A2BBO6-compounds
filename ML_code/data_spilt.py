import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_excel('feature_data_set.xlsx')
# print(df.shape)
train,val=train_test_split(df,train_size=0.85)
train.to_csv('train_data_set.csv',index=False)
val.to_csv('val_data_set.csv',index=False)