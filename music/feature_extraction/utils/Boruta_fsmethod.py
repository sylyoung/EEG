import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from utils import _classification as evaluate
np.random.seed(0)

def load_data():
    # URLS for dataset via UCI
    train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
    train_label_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'

    X_data = pd.read_csv(train_data_url, sep=" ", header=None)
    y_data = pd.read_csv(train_label_url, sep=" ", header=None)
    data = X_data.loc[:, :499]
    data['target'] = y_data[0]
    return data




data = load_data()  # 5 * 501
# print(data.head())
y = data.pop('target')  # label info
X = data.copy().values  # iris data info
print(X.shape, type(y))

# RF classifier
rf = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=7, random_state=0)
# Define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=0)
feat_selector.fit(X, y)
print(type(feat_selector))
# Check selected features
print(feat_selector.support_)
# Select the chosen features from our dataframe.
X_selected = X[:, feat_selector.support_]
y = y.to_numpy()
y = y.squeeze()
print(X_selected.shape, y.shape)  # 特征选择之后的数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# oversampling
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_train_1, X_eval, y_train_1, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
n_0 = 0
n_1 = 0
# for n in range(y.shape[0]):
#     if y[n] == -1:
#         n_0 += 1
#     elif y[n] == 1:
#         n_1 += 1
# ratio = n_0 / n_1
model = XGBClassifier(learning_rate=0.1, n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                      max_depth=6,  # 树的深度
                      min_child_weight=1,  # 叶子节点最小权重
                      gamma=0.,  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,  # 随机选择80%样本建立决策树
                      # colsample_btree=0.8,  # 随机选择80%特征建立决策树
                      objective='binary:logistic',  # 指定损失函数
                      # scale_pos_weight=ratio,  # 解决样本个数不平衡的问题 the ratio of negative and positive fts_labels_r3
                      random_state=42  # 随机数
                      )
model.fit(X_train_1, y_train_1, eval_set=[(X_eval, y_eval)],
              eval_metric="logloss",
              early_stopping_rounds=20,  # 早停点设置
              verbose=False)
y_pred = model.predict(X_test)

try:
    roc_auc = round(roc_auc_score(y_test, y_pred), 3)
except ValueError:
    pass
accuracy = round(accuracy_score(y_test, y_pred), 3)
Sensitivity = round(recall_score(y_test, y_pred, pos_label=1), 3)
specificity = round(evaluate.specificity_score(y_test, y_pred), 3)
print("Auc: %.2f%%, Acc: %.2f%%, Sen: %.2f%%, Spec: %.2f%%" % (
    roc_auc * 100.0, accuracy * 100.0, Sensitivity * 100.0, specificity * 100.0))
# print(feat_selector.ranking_)