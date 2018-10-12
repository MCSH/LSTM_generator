import numpy as np
import os.path
from normalizer import normalizer

def data_load(Config):
    path = Config['data_set']
    text = open(path).read()
    text = normalizer(text)
    print('courpus length:', len(text))

    chars = set(text)
    print('total chars:', len(chars))
    char_indices = dict((c,i) for i,c in enumerate(chars))
    indices_char = dict((i,c) for i,c in enumerate(chars))

    maxlen = 20
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i+maxlen])
        next_chars.append(text[i+maxlen])
    print('nb sequences:', len(sentences))

    print('vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return (chars, char_indices, indices_char, maxlen, X, y, text)
