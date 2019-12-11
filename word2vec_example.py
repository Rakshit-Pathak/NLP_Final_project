# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:12:44 2019

@author: jinseokl
"""

import math
import time
import argparse

import json
import pickle
import nltk

from model import ModelEmbeddings, Network, Hypothesis
import os
import numpy as np
import stanfordnlp  # this is not necessary for loading IMDB dataset

import torch
import torch.nn.utils

import gensim.downloader as api

#stanfordnlp.download('en')   # This downloads the English models for the neural pipeline

preloaded = True

if preloaded == False:
    wv = api.load('word2vec-google-news-300')
    nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English


####### stanfordnlp example usage
#doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
#doc.sentences[0].print_dependencies()
#doc.sentences[1].print_dependencies()

data_sort = False

def load_train_test_imdb_data(data_dir):
    """Loads the IMDB train/test datasets from a folder path.
    Input:
    data_dir: path to the "aclImdb" folder.
    """

    print("... IMDB loading \t\n")
    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), encoding="latin-1") as f:
                    review = f.read()
                    data[split].append([review, score])

    np.random.shuffle(data["train"])        

    return data["train"], data["test"]


if __name__=='__main__':
    #### data loading
    if preloaded == False:
        train_data, test_data = load_train_test_imdb_data(data_dir="./IMDB/aclImdb/")

    print("... data preprocessing starting \t\n")

    num_train = 20
    num_test  = 4
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(num_train):
        X_train.append(np.asarray([wv[word] if word in wv else wv["UNK"] for word in nltk.word_tokenize(train_data[i][0])]))
        Y_train.append(train_data[i][1])
    for i in range(num_test):
        X_test.append(np.asarray([wv[word] if word in wv else wv["UNK"] for word in nltk.word_tokenize(test_data[i][0])]))
        Y_test.append(test_data[i][1])

    data = {'train_x' : X_train,
            'train_y' : Y_train,
            'test_x'  : X_test,
            'test_y'  : Y_test   }
    
    print("saving preprocessed data... \t\n")
    with open('wv_IMDB.pickle', 'wb') as f:
        pickle.dump(data, f)
        f.close()
    
    print("loading preprocessed data... \t\n")
#    with open('wv_IMDB.pickle', 'rb') as f:
#        import_data = pickle.load(f)
#        f.close()

    