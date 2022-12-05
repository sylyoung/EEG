from sklearn.decomposition import PCA
import numpy as np
from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN, BorderlineSMOTE, ADASYN
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from utils import _classification as evaluate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
np.random.seed(0)

warnings.filterwarnings("ignore")

# load feature matrix
file1 = '../data/feature_correct/33_inter_feature.npz'
file2 = '../data/feature_correct/33_ictal_feature.npz'
str_smote = 'SMOTE'  # choose fts_labels_r3 balance approach

test1 = np.load(file1, allow_pickle=True)
test2 = np.load(file2, allow_pickle=True)
print(test1['time'].shape, test1['freq'].shape, test1['tf'].shape, test1['entropy'].shape)
print(test2['time'].shape, test2['freq'].shape, test2['tf'].shape, test2['entropy'].shape)
# joint four class features together
time_fea = np.vstack((test1['time'], test2['time']))  # (n_zero + n_one) * 304 = n_X * 304
freq_fea = np.vstack((test1['freq'], test2['freq']))  # (n_zero + n_one) * 323 = n_X * 323
tf_fea = np.vstack((test1['tf'], test2['tf']))  # (n_zero + n_one) * 627 = n_X * 627
en_fea = np.vstack((test1['entropy'], test2['entropy']))  # (n_zero + n_one) * 152 = n_X * 152
all_fea = np.hstack((time_fea, freq_fea, tf_fea, en_fea))  # n_X * (304 + 323 + 627 + 152) = n_X * 1406

# Create label array according to fts_labels_r3
n_zero = test1['time'].shape[0]
n_one = test2['time'].shape[0]
ratio = n_zero / n_one
y_0 = np.zeros((n_zero, 1))  # 做0类标签
y_1 = np.ones((n_one, 1))  # 做1类标签
y = np.vstack((y_0, y_1))  # 做整体标签

# select the features you want to use for epilepsy detection
X = all_fea[~np.isnan(all_fea).any(axis=1), :]  # or time_fea, freq_fea, tf_fea, en_fea
# check fts_labels_r3
check_nan = np.all(np.isfinite(all_fea))
if check_nan == False:
    # Modify fts_labels_r3
    loca = np.argwhere(np.isnan(all_fea))
    print(loca)  # 获取nan位置
    print(loca.shape[0])
    for i in range(y.shape[0] - X.shape[0]):
        remove = loca[i][0]
        y = np.delete(y, remove - 1)
y = y.squeeze()  # (976,)

# initialize the evaluation for the performance of classify
acc = 0
sen = 0
spec = 0
auc = 0
f1_avg = 0

# 10 folds cross-validation experiment
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # shuffle=True
mrx = np.zeros([2, 2])  # confusion matrix
index = 0
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # scale
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    if str_smote == 'SMOTE':
        # SMOTE
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    elif str_smote == 'SMOTEN':
        # SMOTEN
        oversample = SMOTEN(random_state=42)
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    elif str_smote == 'BLSMOTE':
        # BorderLine SMOTE
        oversample = BorderlineSMOTE(random_state=42)
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    elif str_smote == 'ADASYN':
        # ADASYN
        oversample = ADASYN(random_state=42)
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    elif str_smote == 'NO':
        print('No SMOTE')
    # PCA观察主成分
    test = PCA(500).fit(X_train)  # 约40时包含几乎100%方差
    # 绘图
    # plt.plot(np.cumsum(test.explained_variance_ratio_))
    # plt.xlabel('n components')
    # plt.ylabel('cumulative variance')
    # plt.show()
    # input('')
    # PCA特征选择
    pca = PCA(n_components=800)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # xgb
    model = XGBClassifier(learning_rate=0.1, n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                          max_depth=6,  # 树的深度
                          min_child_weight=1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=0.8,  # 随机选择80%样本建立决策树
                          # colsample_btree=0.8,  # 随机选择80%特征建立决策树
                          objective='binary:logistic',  # 指定损失函数
                          scale_pos_weight=ratio,  # 解决样本个数不平衡的问题 the ratio of negative and positive fts_labels_r3
                          random_state=42  # 随机数
                          )
    X_train_1, X_eval, y_train_1, y_eval = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    model.fit(X_train_1, y_train_1, eval_set=[(X_eval, y_eval)],
              eval_metric="logloss",
              early_stopping_rounds=20,  # 早停点设置
              verbose=False)
    y_pred = model.predict(X_test)
    try:
        roc_auc = round(roc_auc_score(y_test, y_pred), 3)
        auc += roc_auc
    except ValueError:
        auc = auc - 1
        pass
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    acc += accuracy
    Sensitivity = round(recall_score(y_test, y_pred, pos_label=1), 3)
    sen += Sensitivity
    specificity = round(evaluate.specificity_score(y_test, y_pred), 3)
    spec += specificity
    f1 = round(f1_score(y_test, y_pred, pos_label=1), 3)
    f1_avg += f1
    C2 = confusion_matrix(y_test, y_pred, labels=[0, 1])
    mrx += C2
    index += 1
    print('This is the ' + str(index) + ' splits:')
    print('acc:', accuracy)
    print('sen:', Sensitivity)
    print('spec:', specificity)
    print('auc:', roc_auc)
    print('f1_score', f1)
print("Acc_avg: %.2f%%, Sen_avg: %.2f%%, Spec_avg: %.2f%%, Auc_avg: %.2f%%, f1_avg: %.2f%%" % (
    acc * 10.0, sen * 10.0, spec * 10.0, auc * 10.0, f1_avg * 10.0))


