# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:55:52 2020
"""

import pandas as pd
import re

import nltk
import string
nltk.download("wordnet")

'''
Class that contains some basic functions for cleaning data
that is useful for several different transformations
'''


class BaseDataCleaner():
    def __init__(self, raw_text_col, cleaned_text_col):
        self.raw_text_col = raw_text_col
        self.cleaned_col = cleaned_text_col
        
    def fit(self, X):
        return self
        
    def transform(self, X):
        data = X.copy()
        data.loc[:,self.cleaned_col] = self.remove_at_mentions(data, 
                                                         self.raw_text_col)
        data.loc[:,self.cleaned_col] = self.remove_punctuation(data,
                                                         self.cleaned_col)
        data.loc[:,self.cleaned_col] = data[self.cleaned_col].str.lower()
        
        return data[self.cleaned_col].str.split()

    def remove_at_mentions(self, df, data_col):
        return df[data_col].str.replace(re.compile("@[A-Za-z0-9_]+"), "")

    def remove_punctuation(self, df, data_col):
        return df[data_col].str.replace('[{}]'.format(string.punctuation), '')
    