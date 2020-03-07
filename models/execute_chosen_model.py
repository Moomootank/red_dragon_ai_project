# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 06:52:30 2020

"""
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from models.neural_network_builder import NeuralNetworkBuilder
from sklearn.metrics import accuracy_score


def load_pickle(file_loc):
    with open(file_loc, 'rb') as f:
        return pickle.load(f)

def edit_y_to_binary(series):
    copy = series.copy()
    return copy.replace({0:0, 1:0, 2:0, 3:0, 4: 1})

def get_user_predictions(predictions, true_y, user_series):
    pred_df = pd.DataFrame(predictions)
    pred_df['user'] = user_series.values
    mean_user_probabilities = pred_df.groupby('user').mean()
    user_predictions = mean_user_probabilities.idxmax(axis=1)
    pred_df['true_y'] = true_y.values
    user_real_values = pred_df.groupby('user')['true_y'].first()
    return user_predictions, user_real_values
    
        
if __name__ == '__main__':
    # EMBEDDINGS TENSOR NOT PRESENT IN GITHUB AS IT IS TOO BIG
    # Let me know if you really want it and I can send it to you via other means
    embeddings_layer = load_pickle("data/cleaned_data/embeddings_tensor.pickle")
    train_x, train_y, train_users, train_countries = load_pickle("data/cleaned_data/train_data.pickle")
    val_x, val_y, val_users, val_countries = load_pickle("data/cleaned_data/val_data.pickle")
    
    new_train_y = edit_y_to_binary(train_y)
    new_val_y = edit_y_to_binary(val_y)
    nn_builder = NeuralNetworkBuilder()
    # Simple lstm is 0.80 on val set
    simple_lstm = nn_builder.build_simple_lstm_model(embeddings_layer, 
                                                     new_train_y.value_counts().size, 
                                                     128)
    
    simple_lstm.fit(train_x, new_train_y, epochs=6, batch_size=1024)
    simple_lstm.evaluate(val_x, new_val_y, verbose=2)
    predictions_all = simple_lstm.predict(val_x)
    user_preds, user_real_values = get_user_predictions(predictions_all, new_val_y, val_users)
    acc_score =  accuracy_score(user_real_values, user_preds)
    print("Accuracy score is:", accuracy_score(user_real_values, user_preds))
    