from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

# from faker import Faker
import random
from tqdm import tqdm
# from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

from models.mymodel import *

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
print('Yoh.shape = ', Yoh.shape)
outputs = list(Yoh.swapaxes(0,1))
print('Yoh.swapaxes(0,1).shape = ', Yoh.swapaxes(0,1).shape)
print("outputs.len = ", len(outputs))

model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

# model.load_weights('models/model.h5')
#
# EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
#             'March 3rd 2001', '1 March 2001']
#
# for example in EXAMPLES:
#     source = string_to_int(example, Tx, human_vocab)
#     source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
#     prediction = model.predict([source, s0, c0])
#     prediction = np.argmax(prediction, axis=-1)
#     output = [inv_machine_vocab[int(i)] for i in prediction]
#
#     print("source:", example)
#     print("output:", ''.join(output))


# attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)
