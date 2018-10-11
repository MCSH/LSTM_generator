from __future__ import print_function

from keras.callbacks import LambdaCallback

from model import get_model
from data import data_load
from config import Config
from sample import print_sample

(chars, char_indices, indices_char, maxlen, X, y, text) = data_load(Config)

model = get_model(Config, maxlen, chars)

model_name = Config['model_name']

save_iter = Config['save_iter']
print_iter = Config['print_iter']

def on_epoch_end(epoch, _):
    epoch += 1
    print()
    print('Iteration', epoch)

    if epoch % save_iter == 0:
        print ("Saving the model")
        model.save(model_name)

    if epoch % print_iter == 0:
        print_sample(Config, text, maxlen, chars, char_indices, model, indices_char)

    print('-' * 50)

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

epoch_count = Config['epoch_count']
batch_size = Config['batch_size']

model.fit(X, y, batch_size=batch_size, nb_epoch=epoch_count, callbacks= [print_callback])

if Config['save_on_finish']:
    model.save(model_name)
