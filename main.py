import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

from CSP import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings

import mne
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def class4(label_raw, k1, k2):
    label2 = -1
    if k1 < label_raw < k2:
        label2 = label_raw - 1
    elif label_raw > k2:
        label2 = label_raw - 2
    return label2

def get_data_label():
    filepath_label = r'labels_six.mat'
    dict_l = sio.loadmat(filepath_label)
    label1 = dict_l['label_6']
    label = label1 - 1
    label = label.reshape(-1)
    return label

def get_label():
    sign_label = get_data_label()
    label4 = []
    for ii in range(360):
        if sign_label[ii] != 0. and sign_label[ii] != 3.:
            label4.append(class4(sign_label[ii], 0., 3.))
    return label4

def map_to_action_label(number):
    if number == 0:
        return '38'
    elif number == 1:
        return '600'
    elif number == 2:
        return '前'
    elif number == 3:
        return '停'
    else:
        return None

#文件读入字典
eeg_data = sio.loadmat('yj.mat')

#提取字典中的numpy数组
eeg = eeg_data['data']

# 维度信息
num_channels = 127
num_samples = 7000
num_trials = 360
num_channels, num_samples, num_trials = eeg.shape

#获取标签
label6 = get_data_label()
label4 = get_label()
action_labels = np.vectorize(map_to_action_label)(label4)

selected_trials_mask = np.isin(label6, [1, 2, 4, 5])
selected_trials = np.where(selected_trials_mask)[0]

# 提取保留的 trials 对应的 EEG 数据
selected_data = eeg[:, :, selected_trials]

#截取5s正常脑电数据
# 截取每个 trail 中间的 5 秒数据
start_time = (num_samples - 5000) // 2
end_time = start_time + 5000

# 调整数据形状
selected_data = selected_data[:, start_time:end_time, :]

# 将 action_labels 与 selected_data 一一对应
data_with_labels = list(zip(selected_data.transpose(2, 0, 1), action_labels))

# 将数据拆分为特征和标签
features = np.array([data for data, label in data_with_labels])
labels = np.array([label for data, label in data_with_labels])

# 将标签进行编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 标准化数据
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features.reshape(features.shape[0], -1))
scaled_features = scaled_features.reshape(features.shape)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, encoded_labels, test_size=0.3, random_state=42
)
Xtra,Xval,Ytra,Yval = train_test_split(X_train,y_train
                                       ,test_size=0.2
                                       ,random_state=1
                                       )



score_tra_all = []
score_val_all = []
for j in range(10):
    print('num:', j)
    Xtra, Xval, Ytra, Yval = train_test_split(X_train, y_train
                                              , test_size=0.2
                                              , random_state=j * 10
                                              )
    task_left = Xtra[Ytra == 0]
    left_label = Ytra[Ytra == 0]
    task_foot = Xtra[Ytra == 1]
    foot_label = Ytra[Ytra == 1]

    # 训练得到空间滤波器w
    filters = CSP(task_left, task_foot)

    score_tra_one = []
    score_val_one = []
    for i in range(1, 30, 1):
        filters_ = np.concatenate((filters[0][:i], filters[0][-i:]))
        Xtra_csp = []
        for i in range(len(Xtra)):
            Z = np.dot(filters_, Xtra[i])
            Z_csp = np.log(np.var(Z, axis=1) / np.sum(np.var(Z, axis=1)))
            Xtra_csp.append(Z_csp)
        Xtra_csp = np.array(Xtra_csp)
        Xval_csp = []
        for i in range(len(Xval)):
            Z = np.dot(filters_, Xval[i])
            Z_csp = np.log(np.var(Z, axis=1) / np.sum(np.var(Z, axis=1)))
            Xval_csp.append(Z_csp)
        Xval_csp = np.array(Xval_csp)

        clf = SVC(C=0.1, kernel="rbf", cache_size=10000).fit(Xtra_csp, Ytra)
        score_tra = clf.score(Xtra_csp, Ytra)
        score_val = clf.score(Xval_csp, Yval)
        score_tra_one.append(score_tra)
        score_val_one.append(score_val)
    score_tra_one = np.array(score_tra_one)
    score_val_one = np.array(score_val_one)
    score_tra_all.append(score_tra_one)
    score_val_all.append(score_val_one)
score_tra_all = np.array(score_tra_all)
score_val_all = np.array(score_val_all)



score_tra_ave = np.average(score_tra_all,axis=0)
score_val_ave = np.average(score_val_all,axis=0)


filters_num = np.argmax(score_val_ave)



task_left = X_train[y_train==0]   # 74*59*201
left_label = y_train[y_train==0]
task_foot = X_train[y_train==1]    # 66*59*201
foot_label = y_train[y_train==1]
# 训练得到空间滤波器w
filters = CSP(task_left,task_foot)
# 采用上述方法确定的空间滤波器对数
filter_ = np.concatenate((filters[0][:filters_num],filters[0][-filters_num:]))



# 得到CSP特征
Xtrain_csp = []
for i in range(len(X_train)):
    Z = np.dot(filter_,X_train[i])
    Z_csp = np.log(np.var(Z,axis=1)/np.sum(np.var(Z,axis=1)))
    Xtrain_csp.append(Z_csp)
Xtrain_csp = np.array(Xtrain_csp)

Xtest_csp = []
for i in range(len(X_test)):
    Z = np.dot(filter_,X_test[i])
    Z_csp = np.log(np.var(Z,axis=1)/np.sum(np.var(Z,axis=1)))
    Xtest_csp.append(Z_csp)
Xtest_csp = np.array(Xtest_csp)



std_scale = MinMaxScaler().fit(Xtrain_csp)
Xtrain_csp_std = std_scale.transform(Xtrain_csp)
# !!!注意：测试集的归一化也要使用训练集训练的归一化模型std_scale
Xtest_csp_std = std_scale.transform(Xtest_csp)


score_train_c = []
score_test_c = []
for c in np.arange(0.01,1,0.01):
    clf = SVC(C=c,kernel="rbf",cache_size=10000).fit(Xtrain_csp_std,y_train)
    score_train = clf.score(Xtrain_csp_std,y_train)
    score_test = clf.score(Xtest_csp_std,y_test)
    score_train_c.append(score_train)
    score_test_c.append(score_test)
print('c:',c,'score_train:',score_train,'score_test:',score_test)

plt.figure()
plt.plot(np.arange(0.01,1,0.01), score_train_c,label='train acc')
plt.plot(np.arange(0.01,1,0.01), score_test_c,label='test acc')
plt.xlabel('Hyperparameters C')
plt.ylabel('Classification accuacy')
plt.legend()
plt.show();




