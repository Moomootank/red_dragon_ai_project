# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:26:23 2020
"""

import numpy as np
import pandas as pd
import tensorflow as tf

class EmbeddingsMapper():
    def __init__(self, embeddings_url, unknown_token, mask_value):
        self.embeddings = self.load_vectors(embeddings_url)
        self.embeddings.loc[unknown_token] = self.embeddings.mean()
        self.unknown_token = unknown_token
        self.mask_value = mask_value


    def load_vectors(self, vector_url):
        embeddings_dict = {}
        with open(vector_url, 'r', encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], "float32")
                embeddings_dict[word] = vector
        embeddings_dataframe = pd.DataFrame.from_dict(embeddings_dict,
                                                      orient="index")
        toinsert = pd.DataFrame([np.zeros(embeddings_dataframe.shape[1])], 
                                index=["<mask_value>"])
        return pd.concat([toinsert, embeddings_dataframe])

    def create_embeddings_dictionary(self, embeddings_dataframe):
        vocab_list = self.embeddings.index
        word_index_dictionary = {}
        for i in range(len(vocab_list)):
            vocab_word = vocab_list[i]
            word_index_dictionary[vocab_word] = i
        return word_index_dictionary
    
    def map_tokens_to_word_index(self, word_index_dictionary, tokenized_line):
        holder = []
        for word in tokenized_line:
            if word in word_index_dictionary.keys():
                holder.append(word_index_dictionary[word])
            else:
                holder.append(word_index_dictionary[self.unknown_token])
        return holder

    def create_embeddings_tensor(self):
        return tf.keras.layers.Embedding(self.embeddings.shape[0], 
                                         self.embeddings.shape[1],
                                         weights=[self.embeddings],
                                         mask_zero=True)

    def create_padded_input(self, input_value, max_length):
        padded = tf.keras.preprocessing.sequence.pad_sequences(input_value, 
                                                               padding='post',
                                                               maxlen=max_length)
        return padded