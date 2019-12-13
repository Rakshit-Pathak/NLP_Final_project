# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:32:18 2019

@author: jinseokl
"""
import nltk
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors 
import numpy as np

class WVC(object):  # word2vec converter
    def __init__(self, wv_dict=None):
        if wv_dict == None:   self.wv = api.load('word2vec-google-news-300')    # online huge dictionary
        else:                 self.wv = wv_dict                                 # custom dictionary
        
        self.size = len(self.wv)
        
    def word2vec(self, words, is_tokenized=True):
        if not is_tokenized:
            words = nltk.word_tokenize(words)
        return [self.wv[word] if word in self.wv else self.wv["UNK"] for word in words]