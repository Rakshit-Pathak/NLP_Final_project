# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:59:58 2019

@author: Peter
"""

import math
import time
import argparse
import numpy as np
import torch
import torch.nn.utils
import json
import pickle
import matplotlib.pyplot as plt
from models_in_progress.baseline_lstm import LSTM_Model
from utils.preprocessing import WVC

with open('data/trainlist.txt', 'r') as f:
    train_d = json.loads(f.read())


model = LSTM_Model(hidden_size=80)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=.001)

batch_size = 20
num_train_examples = len(train_d)
num_train_batches = int(num_train_examples/batch_size)

loss_function = torch.nn.BCELoss()

with open('../dict_compressed.pickle', 'rb') as f:
    wv_dict = pickle.load(f)
    f.close()
wvc = WVC(wv_dict)

#for batch_index in range(num_train_batches):
for batch_index in range(1):
    batch_data = train_d[batch_index*batch_size:(batch_index+1)*batch_size]
    
    # TODO: convert batch_data to word vectors
    batch_sentences = [batch_data[i][0] for i in range(len(batch_data))]
    batch_class = [batch_data[i][1] for i in range(len(batch_data))]
    batch_sentence_vectors = [wvc.word2vec(sent, is_tokenized=False) for sent in batch_sentences]
    
    optimizer.zero_grad()
    predictions = model(batch_sentence_vectors)
    loss = loss_function(predictions)
    loss.backward()

    # clip gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # using 5.0 as default from HW4

    optimizer.step()