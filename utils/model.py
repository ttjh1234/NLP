import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sentiment_predict(new_sentence,word_index,vocab_size,max_len,model):
    
    '''
    Predict Function for IMDB     
    '''
    
    new_sentence = re.sub('[^0-9a-zA-Z ]', '', new_sentence).lower()
    encoded = []

    # 띄어쓰기 단위 토큰화 후 정수 인코딩
    for word in new_sentence.split():
        try:
            if word_index[word] <= vocab_size:
                encoded.append(word_index[word]+3)
            else:
        # 10,000 이상의 숫자는 <unk> 토큰으로 변환.
                encoded.append(2)
        # 단어 집합에 없는 단어는 <unk> 토큰으로 변환.
        except KeyError:
            encoded.append(2)

    pad_sequence = pad_sequences([encoded], maxlen=max_len)
    print(pad_sequence)
    score = float(model.predict(pad_sequence))

    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))