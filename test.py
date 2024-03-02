import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

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

# 文件读入字典
eeg_data = sio.loadmat('test.mat')

# 提取字典中的numpy数组
eeg = eeg_data['data']

# 维度信息
num_channels = 127
num_samples = 7000
num_trials = 360
num_channels, num_samples, num_trials = eeg.shape

# 获取标签
label6 = get_data_label()
label4 = get_label()
action_labels = np.vectorize(map_to_action_label)(label4)

selected_trials_mask = np.isin(label6, [1, 2, 4, 5])
selected_trials = np.where(selected_trials_mask)[0]

# 提取保留的 trials 对应的 EEG 数据
selected_data = eeg[:, :, selected_trials]

# 截取5s正常脑电数据
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
    scaled_features, encoded_labels, test_size=0.2, random_state=42
)

# 调整数据维度顺序
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(5000, 127)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

# 在测试集上进行预测
y_pred = np.argmax(model.predict(X_test), axis=1)

# 计算准确度和其他评估指标
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
