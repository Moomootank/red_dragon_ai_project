# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:46:27 2020

"""
import pandas as pd
from data_preparers.data_collator import DataCollator
from data_preparers.base_data_cleaner import BaseDataCleaner
from data_preparers.embeddings_mapper import EmbeddingsMapper
from data_preparers.data_splitter import DataSplitter

import pickle

def save_data(to_save, url_to_save):
    with open(url_to_save, 'wb') as f:
        pickle.dump(to_save, f)
        
if __name__ == '__main__':
    collator = DataCollator(["user", "raw_tweet"])
    user_list = collator.get_user_country_list("data/raw_user_lists/" , "Location")
    tweets = collator.combine_all_gathered_tweets("data/gathered_tweets_no_threshold/")
    raw_df = tweets.merge(user_list, left_on="user", right_on="Screen name")
    data = collator.remove_users_below_tweet_threshold(raw_df, "user", 20)
    # For now
    data = data.loc[(data['country']!="singapore") & (data['country']!="malaysia")
                    & (~data['country'].isnull())]
    
    base_cleaner = BaseDataCleaner("raw_tweet", "cleaned_tweet")
    data.loc[:, "cleaned_data"] = base_cleaner.transform(data)
    
    UNKNOWN_TOKEN = "<unk>"
    MASK_TOKEN = "<mask_token"
    em = EmbeddingsMapper("data/glove_twitter_vectors/glove_twitter_100d.txt",
                          UNKNOWN_TOKEN, MASK_TOKEN)
    
    word_index_dictionary = em.create_embeddings_dictionary(em.embeddings)
    data['word_indices'] = data["cleaned_data"].apply(
        lambda x: em.map_tokens_to_word_index(word_index_dictionary, x))
    embeddings_tensor = em.create_embeddings_tensor()
    
    splitter = DataSplitter()
    train_data, val_data, test_data = \
        splitter.split_train_val_test_sets(data, "user", "country", 0.1, 0.2)
    
    MAX_LENGTH = 71
    train_x = em.create_padded_input(train_data["word_indices"], MAX_LENGTH)
    train_y = train_data["trait_number"]
    train_users = train_data["user"]
    save_data((train_x, train_y, train_users, train_data['country']), "data/cleaned_data/train_data.pickle")
    
    val_x = em.create_padded_input(val_data["word_indices"], MAX_LENGTH)
    val_y = val_data["trait_number"]
    val_users = val_data["user"]
    save_data((val_x, val_y, val_users, val_data['country']), "data/cleaned_data/val_data.pickle")
    
    test_x = em.create_padded_input(test_data["word_indices"], MAX_LENGTH)
    test_y = test_data["trait_number"]
    test_users = test_data["user"]
    save_data((test_x, test_y, test_users, test_data['country']), "data/cleaned_data/test_data.pickle")
    
    save_data(embeddings_tensor, "data/cleaned_data/embeddings_tensor.pickle")
    save_data(word_index_dictionary, "data/cleaned_data/word_index_dictionary.pickle")
    