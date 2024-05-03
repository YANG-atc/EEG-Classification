import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import scipy.io as sio

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
eeg_data = sio.loadmat('gz.mat')

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

# 将 action_labels 与 selected_data_middle 一一对应
data_with_labels = list(zip(selected_data.transpose(2, 0, 1), action_labels))

# 将数据拆分为特征和标签
features_middle = np.array([data for data, label in data_with_labels])
labels_middle = np.array([label for data, label in data_with_labels])

# 将标签进行编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels_middle)

# 定义学习率衰减
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# 构建并编译CNN-LSTM模型（简化模型结构以加速训练）
def build_optimized_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=9, strides=2, activation='relu', padding='same', input_shape=input_shape))  # 减少滤波器数量，增加步长
    model.add(Conv1D(256, kernel_size=5, strides=2, activation='relu', padding='same'))  # 减少卷积层，增加步长
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    return model

# 使用优化的学习率
optimizer = Adam(learning_rate=lr_schedule)

# 标准化数据
scaler_middle = StandardScaler()
scaled_features = scaler_middle.fit_transform(features_middle.reshape(features_middle.shape[0], -1))
scaled_features = scaled_features.reshape(features_middle.shape)

# 调整数据形状以匹配CNN-LSTM模型的输入
input_shape = (scaled_features.shape[2], scaled_features.shape[1])  # 调整为(time_steps, channels)
scaled_features_reshaped = scaled_features.transpose((0, 2, 1))

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features_reshaped, encoded_labels, test_size=0.2, random_state=42
)

# 构建模型并编译
model = build_optimized_cnn_lstm_model(input_shape)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 添加早停回调
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# 训练模型，增加批量大小
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=128, callbacks=[early_stopping])


# 构建并编译CNN-LSTM模型
model = build_optimized_cnn_lstm_model(input_shape)
model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=256)

# 在测试集上进行预测
y_pred = np.argmax(model.predict(X_test), axis=1)

# 计算准确度和其他评估指标
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')