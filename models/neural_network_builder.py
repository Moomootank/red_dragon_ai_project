# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:30:09 2020

"""
import tensorflow as tf
from tensorflow.keras import layers

class NeuralNetworkBuilder():

    def build_simple_lstm_model(self, embeddings_layer, num_classes, 
                                num_neurons):
        model = tf.keras.Sequential()
        model.add(embeddings_layer)
        model.add(layers.LSTM(num_neurons, activation="relu"))
        model.add(layers.Dense(num_classes))
        model.add(layers.Softmax())
        print(model.summary())
        
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', 
                      loss=loss_function,
                      metrics=["sparse_categorical_accuracy"])
                      
        return model
