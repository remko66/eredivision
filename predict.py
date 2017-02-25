import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.optimizers import RMSprop
from keras.utils import np_utils

def model():
    global dict
    model = Sequential()
    model.add(Dense(200, input_shape=(2,len(dict),)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.7))
    model.add(Dense(60))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

def one_to_hot(team):
    global dict
    ar=np.zeros((len(dict)), dtype=np.uint)
    teamnr=dict[team]
    ar[teamnr]=1
    return ar
def one_hot_res(result):
    ar = np.zeros(3, dtype=np.uint)
    if(result=='H'):
        ar[0]=1
    if (result == 'D'):
        ar[1] = 1
    if (result == 'A'):
        ar[2] = 1
    return ar


dict={}
dict_rev={}
df = pd.read_csv('data/N1.csv')
dx = pd.read_csv('data/fixtures.csv')

df.drop(df.columns[[0,1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]], axis=1, inplace=True)
dx.drop(dx.columns[[0,1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]], axis=1, inplace=True)
for i,t in enumerate(df['HomeTeam']):
    if t not in dict:
        dict[t]=i
        dict_rev[i]=t
x=np.zeros((len(df),2,len(dict)), dtype=np.uint)
y=np.zeros((len(df),3),dtype=np.float32)
x_pred=np.zeros((len(dx),2,len(dict)),dtype=np.uint)
for a,b in df.iterrows():
    home=b[0]

    away=b[1]
    x[a,0,:]=one_to_hot(home)
    x[a,1, :] = one_to_hot(away)
    result=b[4]
    y[a,:]=one_hot_res(result)
fix=[]
count=0
for a, b in dx.iterrows():
    home = b[0]
    away = b[1]
    if home in dict:
        count+=1
x_pred=np.zeros((count,2,len(dict)),dtype=np.uint)
count=0
for a, b in dx.iterrows():
    home = b[0]
    away = b[1]
    if home in dict:
        fix.append(home + '-' + away)
        x_pred[count,0,:]=one_to_hot(home)
        x_pred[count, 1, :] = one_to_hot(away)
        count+=1

model=model()
print(model)

model.fit(x, y, batch_size=1, nb_epoch=50,
          verbose=1)
pred=model.predict(x_pred)
print(pred.shape,len(fix))
print("match","home","draw","away")
for a,p in enumerate(pred):
    home=round(p[0]*100)
    draw=round(p[1]*100)
    away=round(p[2]*100)
    print(fix[a],home,draw,away)