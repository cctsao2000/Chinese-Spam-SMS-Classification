import numpy as np 
from sklearn import metrics
from sklearn.model_selection import train_test_split
with open('train_X.npy', 'rb') as tr_X:
  train_X = np.load(tr_X)
with open('test_X.npy', 'rb') as te_X:
  test_X = np.load(te_X)
with open('train_label.npy', 'rb') as tr_y:
  train_label = np.load(tr_y)
with open('test_label.npy', 'rb') as te_y:
  test_label = np.load(te_y)
  train_X_, val_X, train_label_, val_label = train_test_split(train_X, train_label, test_size=.2, stratify=train_label, random_state=42)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# 增加模型層數與修改模型設計
model = Sequential()
model.add(Embedding(10000, 128))
model.add(LSTM(128))
# 增加Dropout層，防止過擬和
model.add(Dropout(0.2))
# 全連接層
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='SGD',metrics=['accuracy'])
model.summary()
from keras.callbacks import EarlyStopping   # 利用Early Stopping方法節省訓練時間，如果在3個epoch內沒提升模型表現就提前終止訓練（防止花了時間訓練卻過擬和）
from keras.callbacks import ModelCheckpoint # 利用Model Checkpoint將目前為止訓練的最佳模型暫存起來

callback = [EarlyStopping(monitor='val_loss', patience=3, mode='max'),
            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(train_X_, train_label_, batch_size=64, epochs=10, callbacks=[callback],validation_data=(val_X, val_label))