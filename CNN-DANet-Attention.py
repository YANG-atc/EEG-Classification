import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.layers import Input, MaxPooling2D, Conv2D
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
def class4 (label_raw, k1, k2):
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
        return '未知'

# 文件读入字典
eeg_data = sio.loadmat('jc.mat')

# 提取字典中的numpy数组
eeg = eeg_data['data']

# 维度信息
num_channels = 127
num_samples = 7000  # 修改为实际数据的总样本点数
num_trials = 360
num_channels, num_samples, num_trials = eeg.shape

# 获取标签
label6 = get_data_label()
label4 = get_label()
action_labels = np.vectorize(map_to_action_label, otypes=[str])(label4)

selected_trials_mask = np.isin(label6, [1, 2, 4, 5])
selected_trials = np.where(selected_trials_mask)[0]

# 提取保留的 trials 对应的 EEG 数据
selected_data = eeg[:, :, selected_trials]

# 截取5s正常脑电数据
# 截取每个 trial 中间的 5 秒数据
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
    scaled_features, encoded_labels, test_size=0.1, random_state=42
)

def channel_attention(inputs, ratio=8,l2_reg=1e-5):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    channel_dim = int(inputs.shape[channel_axis])

    # 平均池化和最大池化
    avg_pool = tf.reduce_mean(inputs, axis=channel_axis, keepdims=True)
    max_pool = tf.reduce_max(inputs, axis=channel_axis, keepdims=True)

    # 共享的全连接层，确保滤波器数量至少为1
    reduced_dim = max(channel_dim // ratio, 1)
    squeeze = tf.concat([avg_pool, max_pool], axis=channel_axis)

    fc1 = tf.keras.layers.Conv1D(reduced_dim, 1, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(squeeze)
    fc2 = tf.keras.layers.Conv1D(channel_dim, 1, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(fc1)

    # 重缩放特征
    scale = inputs * fc2
    return scale


def temporal_attention(inputs, l2_reg=1e-5):
    # 使用全局平均池化和最大池化得到时序上的重要性
    avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

    # 合并并使用一个小型的卷积层来获得权重

    attention_weights = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention_weights = tf.keras.layers.Conv1D(1, kernel_size=1, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(attention_weights)

    # 应用注意力权重
    scaled_inputs = inputs * attention_weights
    return scaled_inputs

input_shape = (num_channels, 5000, 1)
inputs = Input(shape=input_shape)


# 应用danet到输入数据
# 假设输入是四维张量，我们先进行维度调整，以便处理
def adjust_dimensions(inputs):
    # 这里假设第2维（127）是通道维度，需要转换为一维卷积兼容的格式
    # 例如，如果是通道优先（channels_first），则调整顺序
    permuted_inputs = tf.keras.layers.Permute((2, 3, 1))(inputs)  # 假定第1维是时间步长，第3维是通道
    return permuted_inputs

# 应用维度调整
adjusted_inputs = adjust_dimensions(inputs)
print("Adjusted Inputs", adjusted_inputs.shape)

# 现在，我们假设调整后可以直接应用channel_attention，但注意，这里需要根据实际维度调整channel_axis
channel_attended = channel_attention(adjusted_inputs)
print("channel_attended", channel_attended.shape)

# 对channel_attended进行必要的逆调整，以便与temporal_attention兼容
# 注意，这一步取决于channel_attended输出的形状，这里仅示意
reversed_adjusted = tf.keras.layers.Permute((3, 1, 2))(channel_attended)
temporal_attended = temporal_attention(reversed_adjusted)
print("temporal_attention", temporal_attended.shape)

# 添加全局平均池化层（Global Average Pooling, GAP）
adjusted_temporal = tf.keras.layers.Permute((2, 1, 3))(temporal_attended)
adjusted_temporal = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(adjusted_temporal)
print("Adjusted Temporal Attention After Squeeze", adjusted_temporal.shape)

# 现在形状应该是(None, 5000, 127)，可以应用GlobalAveragePooling1D
# 添加全局平均池化层（Global Average Pooling, GAP）
gap_layer = layers.GlobalAveragePooling1D()(adjusted_temporal)

# 在Dense层之前加入dropout
dropout_layer = Dropout(rate=0.5)  # 选择合适的丢弃比例，如0.5
gap_dropout = dropout_layer(gap_layer)
additional_fc = layers.Dense(64, activation='relu')(gap_dropout)  # 新增的全连接层
dropout_additional_fc = Dropout(rate=0.2)(additional_fc)  # 为新增层也添加Dropout
# 添加全连接层（Dense layer）以输出最终的类别预测
# 假设有4个类别，因此最后一个神经元的数量应该是4
predictions = layers.Dense(4, activation='softmax')(dropout_additional_fc)

# 构建模型
model = Model(inputs=inputs, outputs=predictions)

# 参数定义
batch_size = 32
num_epochs = 30
num_batches_per_epoch = X_train.shape[0] // batch_size


# 使用ExponentialDecay
initial_learning_rate = 0.001
decay_rate = 0.96
decay_steps = num_batches_per_epoch * (num_epochs // 10)  # 每10个epoch衰减
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate,
    staircase=True
)

# 初始化Adam优化器并设置动态学习率
optimizer = Adam(learning_rate=lr_schedule)

# 定义EarlyStopping回调
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 定义ReduceLROnPlateau回调
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# 组合回调列表
callbacks_list = [early_stopping, reduce_lr]

# 编译模型
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 模型训练
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    validation_split=0.1,
                    callbacks=callbacks_list)

# 测试集评估
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)