# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:12:44 2019

@author: jinseokl
"""

import math
import time
import argparse
import pickle
import nltk
import json
import os
import numpy as np
import torch
import torch.nn.utils


from preprocessing import WVC


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


preloaded = False

if __name__=='__main__':
    """
    1) load predefined w2v dictionary
    2) load word2vec converter (WVC from preprocessing)
    3) use function WVC.word2vec(words, is_tokenized=True),
        it returns a numpy array with the size of (sentence length, embedding size=300)
    """
    
    #### data loading
    if not preloaded:
        with open('trainlist.txt', 'r') as f:
            data_train = json.load(f)
            f.close()
            
        with open('trainlist.txt', 'r') as f:
            data_train = json.load(f)
            f.close()
    
        with open('dict_compressed.pickle', 'rb') as f:
            wv_dict = pickle.load(f)
            f.close()

    ###### preprocessing example
    wvc = WVC(wv_dict)
    
    example_sentence = data_train[0][0]
    
    example_vector1 = wvc.word2vec(example_sentence, is_tokenized=False)    
    example_vector2 = wvc.word2vec(nltk.word_tokenize(example_sentence))
    example_vector3 = wvc.word2vec(nltk.word_tokenize(example_sentence), is_tokenized=True)  # same as example2

    

=======
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:12:44 2019

@author: jinseokl
"""

import math
import time
import argparse
import pickle
import nltk
import json
import os
import numpy as np
import torch
import torch.nn.utils


from preprocessing import WVC


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


preloaded = False

if __name__=='__main__':
    """
    1) load predefined w2v dictionary
    2) load word2vec converter (WVC from preprocessing)
    3) use function WVC.word2vec(words, is_tokenized=True),
        it returns a numpy array with the size of (sentence length, embedding size=300)
    """
    
    #### data loading
    if not preloaded:
        with open('trainlist.txt', 'r') as f:
            data_train = json.load(f)
            f.close()
            
        with open('trainlist.txt', 'r') as f:
            data_train = json.load(f)
            f.close()
    
        with open('dict_compressed.pickle', 'rb') as f:
            wv_dict = pickle.load(f)
            f.close()

    ###### preprocessing example
    wvc = WVC(wv_dict)
    
    example_sentence = data_train[0][0]
    
    example_vector1 = wvc.word2vec(example_sentence, is_tokenized=False)    
    example_vector2 = wvc.word2vec(nltk.word_tokenize(example_sentence))
    example_vector3 = wvc.word2vec(nltk.word_tokenize(example_sentence), is_tokenized=True)  # same as example2

    
    batch_size = 10
    sent_batch = [data_train[i][0] for i in range(10)]
    example_vector_batch = [wvc.word2vec(sent, is_tokenized=False) for sent in sent_batch]
    
    