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
num_train_batches = int(num_train_examples/batch_size)

loss_function = torch.nn.BCELoss()

# split apart the train_text and train_labels
train_text,train_labels = zip(*data_train)

print_every = 5
batch_loss_avg = 0

for batch_index in range(num_train_batches):
    
    batch_reviews_text = train_text[batch_index*batch_size:(batch_index+1)*batch_size]
    batch_labels = train_labels[batch_index*batch_size:(batch_index+1)*batch_size]
    
    # embedding the text
    batch_reviews_vec = [wvc.word2vec(review, is_tokenized=False) for review in batch_reviews_text]
    
    # sort the batch sentences by length (will be necessary for pack_padded_sequence)
    review_lens = [-len(review) for review in batch_reviews_vec]
    sort_inds = np.argsort(review_lens)
    batch_reviews_vec_sorted = [batch_reviews_vec[i] for i in sort_inds]
    batch_labels_sorted = [batch_labels[i] for i in sort_inds]

    optimizer.zero_grad()
    predictions = model(batch_reviews_vec_sorted)
    
    loss = loss_function(torch.squeeze(predictions),torch.tensor(batch_labels_sorted, dtype=torch.float))
    loss.backward()

    batch_loss = loss.sum()
    batch_losses_val = batch_loss.item()
    batch_loss_avg += batch_losses_val/print_every
    
    # clip gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # using 5.0 as default from HW4

    optimizer.step()
    
    if batch_index % print_every == 0:
        print('Batch '+str(batch_index)+'/'+str(num_train_batches)+' loss: '+str(batch_loss_avg))
        batch_loss_avg = 0
    