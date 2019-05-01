from model.RNNModel import *

data = open('data/dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))

char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

model(data, ix_to_char, char_to_ix)