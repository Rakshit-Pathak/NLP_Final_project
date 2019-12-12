# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:26:29 2019

@author: jinseokl
"""
import nltk
import json
import pickle
import gensim.downloader as api
import stanfordnlp  # this is not necessary for loading IMDB dataset


#wv = api.load('word2vec-google-news-300')      # uncomment if none of word2vec dictionaary is loaded

#################################
with open('trainlist.txt', 'r') as f:
    data_train = json.load(f)
    f.close()
    
with open('testlist.txt', 'r') as f:
    data_test = json.load(f)
    f.close()
    
#################################
dict_compressed = {}

print("train data dictionary generation")
for i in range(len(data_train)):
#for i in range(5):
     tokens = nltk.word_tokenize(data_train[i][0])
     for word in tokens:
         if word in wv:
             dict_compressed[word] = wv[word]
         else:
             dict_compressed[word] = wv["UNK"]

print("test data dictionary generation")             
for i in range(len(data_test)):
     tokens = nltk.word_tokenize(data_train[i][0])
     for word in tokens:
         if word in wv:
             dict_compressed[word] = wv[word]
         else:
             dict_compressed[word] = wv["UNK"]

dict_compressed["UNK"] = wv["UNK"]


################################
print("saving dictionary")                          
with open('dict_compressed.pickle','wb') as f:
    pickle.dump(dict_compressed,f)
    f.close()
    
print("loading dictionary")    
with open('dict_compressed.pickle','rb') as f:
    dict_in = pickle.load(f)
    f.close()