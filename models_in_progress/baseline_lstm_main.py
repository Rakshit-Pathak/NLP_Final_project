# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:59:58 2019

@author: Peter
"""

import math
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn.utils
import json
import matplotlib.pyplot as plt
from baseline_lstm import LSTM_Model

from baseline_lstm import pad_sents

import sys
sys.path.append('../') # needed to access functions in the parent directory

from preprocessing import WVC

with open('../trainlist.txt', 'r') as f:
    data_train = json.loads(f.read())

with open('dict_compressed.pickle', 'rb') as f:
    wv_dict = pickle.load(f)
    f.close()

###### preprocessing 
wvc = WVC(wv_dict)

model = LSTM_Model(hidden_size=80)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=.001)

batch_size = 20
num_train_examples = len(data_train)
num_train_batches = 1 #int(num_train_examples/batch_size)

loss_function = torch.nn.BCELoss()

# split apart the train_text and train_labels
train_text,train_labels = zip(*data_train)

for batch_index in range(num_train_batches):
    
    batch_reviews_text = train_text[batch_index*batch_size:(batch_index+1)*batch_size]
    batch_labels = train_labels[batch_index*batch_size:(batch_index+1)*batch_size]
    
    batch_reviews_vec = [wvc.word2vec(review, is_tokenized=False) for review in batch_reviews_text]
    batch_reviews_vec = pad_sents(batch_reviews_vec,0)
    batch_reviews_vec = np.stack(batch_reviews_vec, axis=1)  # creates np array of shape [max_T, batch_size, embedded_dim]
    
    optimizer.zero_grad()
    predictions = model(batch_reviews_vec)
    loss = loss_function(predictions,np.asarray(batch_labels))
    loss.backward()

    # clip gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # using 5.0 as default from HW4

    optimizer.step()