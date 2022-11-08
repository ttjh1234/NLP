import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sentence_decode(id_to_word,encode_data):
    sentence=[]
    for i in encode_data:
        if i not in [0,1,2,3]:
            sentence.append(id_to_word[i])

    print(' '.join(sentence))