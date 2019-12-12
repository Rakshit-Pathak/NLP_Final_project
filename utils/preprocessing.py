# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:32:18 2019

@author: jinseokl
"""
import nltk
import pickle
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors 
import numpy as np

class WVC(object):  # word2vec converter
    def __init__(self, wv_dict=None):
        print('=== word2vec converter (WVC) initializing...')
        if wv_dict == 'google'  :  self.wv = api.load('word2vec-google-news-300')    # online huge dictionary
        elif wv_dict == 'empty' :  self.wv = {}
        elif wv_dict == None    :  self.wv = {}
        else                    :  self.wv = wv_dict                                 # custom dictionary        
        
    
    def __len__(self):
        self.size = len(self.wv)
        return self.size
        
    
    def remove_dict(self):
        self.wv = {}
    
    
    def word2vec(self, words, is_tokenized=True):
        if not is_tokenized:
            words = nltk.word_tokenize(words)
        return np.asarray([self.wv[word] if word in self.wv else self.wv["UNK"] for word in words])
    
    
    def add_dict_sent(self, sent, wv_ref, is_tokenized=True):
        """
            sent         : input sentence that will be added to the current dictionary
            wv_ref       : reference word2vec dictionary (eg. word2vec-google-news-300 )
            is_tokenized : flag indicating sentence tokenized 
        """
        if not is_tokenized:   sent = nltk.word_tokenize(sent)
        for word in sent:
            self.wv[word] = wv_ref[word]
    
        
    def save_dict(self, dest):
        print('=== start saving wv dictionary...')
        with open(dest, 'wb') as f:
            pickle.dump(self.wv, f)
            f.close()
        print('=== done saving wv dictionary...')
    
    
    def load_dict(self, predef_dict):
        print('=== start loading wv dictionary...')
        with open(predef_dict, 'rb') as f:
            self.wv = pickle.load(f)
            f.close()
        print('=== done loading wv dictionary...')
            