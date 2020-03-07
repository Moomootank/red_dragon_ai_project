# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:19:09 2020

@author: Chia Wei Jie
"""
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter():
    def split_train_val_test_sets(self, df, user_column, trait_column,
                             val_size, test_size):
        df_copy = df.copy()
        TRAIT_NUM_COL = "trait_number"
        df_copy[TRAIT_NUM_COL] = df_copy[trait_column].astype("category").cat.codes

        user_trait_df = df_copy.groupby(user_column)[TRAIT_NUM_COL].first()
        train_val_users, test_users = train_test_split(user_trait_df, 
                                                       test_size=test_size,
                                                       stratify=user_trait_df)
        train_users, val_users = train_test_split(train_val_users,
                                                  test_size=val_size,
                                                  stratify=train_val_users)
        train_users = train_users.index.values

        train_data = df_copy.loc[df_copy[user_column].isin(train_users)]
        val_users = val_users.index.values
        val_data = df_copy.loc[df_copy[user_column].isin(val_users)]
        test_users = test_users.index.values
        test_data = df_copy.loc[df_copy[user_column].isin(test_users)]
        
        return train_data, val_data, test_data
