from sklearn.manifold import Isomap
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, SMOTEN, BorderlineSMOTE, ADASYN
import os
def seed_everything(seed=42):
    """"
    Seed everything.
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
# load feature matrix
file1 = '../data/feature_correct/34_inter_feature.npz'
file2 = '../data/feature_correct/34_ictal_feature.npz'
str_smote = 'SMOTE'  # choose data balance approach

test1 = np.load(file1, allow_pickle=True)
test2 = np.load(file2, allow_pickle=True)
n0 = test1['time'].shape[0]
n1 = test2['time'].shape[0]
# joint four class features together
time_fea = np.vstack((test1['time'], test2['time']))  # (n_zero + n_one) * 304 = n_X * 304
freq_fea = np.vstack((test1['freq'], test2['freq']))  # (n_zero + n_one) * 323 = n_X * 323
tf_fea = np.vstack((test1['tf'], test2['tf']))  # (n_zero + n_one) * 627 = n_X * 627
en_fea = np.vstack((test1['entropy'], test2['entropy']))  # (n_zero + n_one) * 152 = n_X * 152
all_fea = np.hstack((time_fea, freq_fea, tf_fea, en_fea))  # n_X * (304 + 323 + 627 + 152) = n_X * 1406

# 导入数据
data = all_fea  # (n, fts_num)
# 构造标签
y_0 = np.zeros((n0, 1))  # 做0类标签
y_1 = np.ones((n1, 1))  # 做1类标签
y = np.vstack((y_0, y_1))  # 做整体标签
y = y.squeeze()

# IsoMap 流形方法
model = Isomap(n_neighbors=10, n_components=2)  # 对所有数据的二维投影
projection = model.fit_transform(data)
# 画图
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
            c=y, cmap=plt.cm.get_cmap('Accent', 2))
plt.colorbar(ticks=range(2), label='digit_value')
plt.show()

# smote处理
oversample = SMOTE(random_state=42)
data_smote, y = oversample.fit_resample(data, y)
print(data_smote.shape, y.shape)
# IsoMap 流形方法
model = Isomap(n_neighbors=10, n_components=2)  # 对所有数据的二维投影
test = model.fit_transform(data_smote)
# 画图
plt.scatter(test[:, 0], test[:, 1], lw=0.1,
            c=y, cmap=plt.cm.get_cmap('Accent', 2))
plt.colorbar(ticks=range(2), label='digit_value')
plt.show()
