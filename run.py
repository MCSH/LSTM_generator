from __future__ import print_function

from keras.callbacks import LambdaCallback

from model import get_model
from data import data_load
from config import Config
from sample import print_sample

(chars, char_indices, indices_char, maxlen, X, y, text) = data_load(Config)

model = get_model(Config, maxlen, chars)

model_name = Config['model_name']

print_sample(Config, text, maxlen, chars, char_indices, model, indices_char)
