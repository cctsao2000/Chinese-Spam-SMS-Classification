import numpy as np 
import tensorflow as tf 
from sklearn.metrics import roc_auc_score
from sklearn import metrics

with open('train_X.npy', 'rb') as tr_X:
    train_X = np.load(tr_X)
with open('test_X.npy', 'rb') as te_X:
    test_X = np.load(te_X)
with open('train_label.npy', 'rb') as tr_y:
    train_label = np.load(tr_y)
with open('test_label.npy', 'rb') as te_y:
    test_label = np.load(te_y)

print(train_X.shape)
print(test_X.shape)
print(train_label.shape)
print(test_label.shape)


# model = LogisticRegression()
# model.fit(train_X, train_label)
# pred = model.predict(test_X)

# train_acc = model.score(train_X, train_label)
# print("Train accuracy: ", train_acc)
# test_acc = model.score(test_X, test_label)
# print("Test accuracy: ", test_acc)

# pred_prob_y = model.predict_proba(test_X)[:, 1]
# test_auc = roc_auc_score(test_label, pred_prob_y)
# print("Test AUC: ", test_auc)