import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def imdb_load(vocab_size=10000,max_len=500):

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    word_index = tf.keras.datasets.imdb.get_word_index()
    id_to_word={id_ + 3 : word for word,id_ in word_index.items()}

    for id_,token in enumerate(["<pad>","<sos>","<unk>"]):
        id_to_word[id_]=token

    return (X_train,y_train),(X_test,y_test),word_index,id_to_word


