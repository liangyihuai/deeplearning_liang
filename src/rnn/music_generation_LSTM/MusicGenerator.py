from __future__ import print_function
from LSTMModel import *
import IPython
from data_utils import *
from keras.optimizers import Adam

print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

# 查看输入和输出的数据集，它们都被表示成了one-hot的形式
print("X[0, :, :] = ", X[0, :, :])
print("Y[0, :, :] = ", Y[0, :, :])

# 构建模型
model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
# 指定Adam优化器
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
# 编译模型
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
model.fit([X, a0, c0], list(Y), epochs=100)

# 构建用于生成音乐的模型，该模型与“训练模型”是不一样的
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)

# 生成音乐，所生成的音乐将存储在./output文件下面
out_stream = generate_music(inference_model)