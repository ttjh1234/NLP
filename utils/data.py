import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torchtext
from torchtext.datasets import IMDB


def imdb_load(vocab_size=10000,max_len=500):

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    word_index = tf.keras.datasets.imdb.get_word_index()
    id_to_word={id_ + 3 : word for word,id_ in word_index.items()}

    for id_,token in enumerate(["<pad>","<sos>","<unk>"]):
        id_to_word[id_]=token

    return (X_train,y_train),(X_test,y_test),word_index,id_to_word

def imdb_pytorch_load(vocab_size=10000,max_len=500,batch_size=64):
    TEXT = torchtext.data.Field(sequential=True, lower=True, batch_first=True,fix_length=max_len)
    LABEL = torchtext.data.Field(sequential=False,batch_first= True)
    trainset, testset = torchtext.datasets.IMDB.splits(TEXT, LABEL)
    
    TEXT.build_vocab(trainset, min_freq=5,max_size=vocab_size) # 단어 집합 생성
    LABEL.build_vocab(trainset)
    
    vocab_size = len(TEXT.vocab)
    n_classes = 2
    print('단어 집합의 크기 : {}'.format(vocab_size))
    print('클래스의 개수 : {}'.format(n_classes))

    trainset, valset = trainset.split(split_ratio=0.8)
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=batch_size, shuffle=True, repeat=False)
    
    print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iter)))
    print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iter)))
    print('검증 데이터의 미니 배치의 개수 : {}'.format(len(val_iter)))
    
    input_dim=len(TEXT.vocab)
    TEXT_PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    return train_iter, val_iter, test_iter, input_dim, TEXT_PAD_IDX, TEXT