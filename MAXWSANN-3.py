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

from keras.layers import Dense, activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Dense(78, input_dim=78, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
print(model.summary())

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

history = model.fit(X_train_stand, y_train_stand, validation_split = 0.21, epochs = 50, batch_size=64)
y_pred = model.predict(X_test_stand)

OrigPred = y_pred*1000

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test_stand, y_pred))

out = pd.DataFrame(OrigPred)
print(out)
out.to_excel('output_ANN.xlsx', index=False)
input('Press Enter to STOP')