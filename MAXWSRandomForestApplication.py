import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('ProjectDataMAXWS-3.xlsx')
final_df = pd.get_dummies(df, columns=['City','Storm Month','WD (name)','WD (name)1','WD (name)2','WD (name)3'])

X = final_df.drop(['Percentage Customers Affected'], axis=1)
y = final_df['Percentage Customers Affected']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.21, random_state=0)

X_train_stand = X_train.copy()
X_test_stand = X_test.copy()

num_cols = ['WS','WG','WS1','WG1','WS2','WG2','WS3','WG3','Storm Time (hrs.)']

for i in num_cols:
    scale = StandardScaler().fit(X_train_stand[[i]])
    X_train_stand[i] = scale.transform(X_train_stand[[i]])
    X_test_stand[i] = scale.transform(X_test_stand[[i]])

from sklearn import preprocessing

y_train_stand = preprocessing.normalize([y_train])
y_train_stand = y_train_stand.T

y_test_stand = preprocessing.normalize([y_test])
y_test_stand = y_test_stand.T

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train_stand,y_train_stand)

pred = rf.predict(X_test_stand)
OrigPred = pred*1000

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(pred, y_test_stand))

AppData = pd.read_excel('26thJan.xlsx')
Weather = AppData.drop(['Percentage Customers Affected'], axis=1)
weather_copy = Weather.copy()
Damage = AppData['Percentage Customers Affected']


num_cols = ['WS','WG','WS1','WG1','WS2','WG2','WS3','WG3','Storm Time (hrs.)']

for i in num_cols:
    scale = StandardScaler().fit(weather_copy[[i]])
    weather_copy[i] = scale.transform(weather_copy[[i]])

damage_pred = rf.predict(weather_copy)
OrigDamagePred = damage_pred*1000
DamOut = pd.DataFrame(OrigDamagePred, columns=['Percentage Customers Affected'])
print(DamOut)

DamOut.to_excel('DamOut.xlsx', index=False)
input('Press Enter to STOP')