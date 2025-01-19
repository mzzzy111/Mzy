import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
import matplotlib as mpl

warnings.filterwarnings("ignore")
np.random.seed(120)
tf.random.set_seed(120)

# 设置路径复杂度限制
mpl.rcParams['agg.path.chunksize'] = 10000

# 设置 GPU 可见性
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# 加载数据
data = pd.read_csv(r"D:\code\pycode\trace_prediction-main\taxi1.csv")

# 数据预处理
def createSequence(data, window=10, maxmin=None):
    train_seq = []
    train_label = []
    m, n = maxmin
    for trip_id in range(len(data) - window):
        data_temp = data.iloc[trip_id:trip_id + window]
        data_temp = np.array(data_temp.loc[:, ['PULocationID', 'DOLocationID', 'trip_distance', 'fare_amount']])
        # 标准化
        data_temp = (data_temp - n) / (m - n)

        x = data_temp[:-1]
        y = data_temp[-1]
        train_seq.append(x)
        train_label.append(y)

    train_seq = np.array(train_seq, dtype='float64')
    train_label = np.array(train_label, dtype='float64')

    return train_seq, train_label

# 计算归一化参数
nor = np.array(data)
m = nor.max(axis=0)
n = nor.min(axis=0)
maxmin = [m, n]
# 步长
windows = 10
train_seq, train_label = createSequence(data, windows, maxmin)

# 划分训练集和测试集
split = int(0.8 * len(train_seq))
train_X, test_X = train_seq[:split], train_seq[split:]
train_Y, test_Y = train_label[:split], train_label[split:]

# 创建和训练模型
def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation("relu"))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    log = CSVLogger(f"./log.csv", separator=",", append=True)
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)

    model.fit(train_X, train_Y, epochs= 5,batch_size=32, verbose=1, validation_split=0.1,
                  callbacks=[log, reduce])
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    model.save(f"./model.h5")
    model.summary()
    return model

# 训练模型
model = trainModel(train_X, train_Y, test_X, test_Y)

# 预测和绘图
def FNormalizeMult(y_pre, y_true, max_min):
    [m1, n1, d1, f1], [m2, n2, d2, f2] = max_min
    y_pre[:, 0] = y_pre[:, 0] * (m1 - m2) + m2
    y_pre[:, 1] = y_pre[:, 1] * (n1 - n2) + n2
    y_pre[:, 2] = y_pre[:, 2] * (d1 - d2) + d2
    y_pre[:, 3] = y_pre[:, 3] * (f1 - f2) + f2
    y_true[:, 0] = y_true[:, 0] * (m1 - m2) + m2
    y_true[:, 1] = y_true[:, 1] * (n1 - n2) + n2
    y_true[:, 2] = y_true[:, 2] * (d1 - d2) + d2
    y_true[:, 3] = y_true[:, 3] * (f1 - f2) + f2

    return y_pre, y_true

# 示例预测和绘图
y_pre = model.predict(test_X)
f_y_pre, f_y_true = FNormalizeMult(y_pre, test_Y, maxmin)

# 采样数据以减少数据点
sample_rate = 100 # 每10个点取一个
f_y_pre_sampled = f_y_pre[::sample_rate]
f_y_true_sampled = f_y_true[::sample_rate]

plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.plot(f_y_pre[:, 0], f_y_pre[:, 1], "b-", label='预测值')
plt.plot(f_y_true[:, 0], f_y_true[:, 1], "r-", label='真实值')
plt.legend(fontsize=14)
plt.title("出租车轨迹预测", fontsize=17)
plt.xlabel("PULocationID", fontsize=14)
plt.ylabel("DOLocationID", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()

plt.subplot(122)
plt.plot(f_y_pre[:, 2], "b-", label='预测值')
plt.plot(f_y_true[:, 2], "r-", label='真实值')
plt.legend(fontsize=14)
plt.title("行程距离预测", fontsize=17)
plt.xlabel("时间步", fontsize=14)
plt.ylabel("行程距离", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()

plt.show()