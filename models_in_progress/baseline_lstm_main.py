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
import matplotlib.pyplot as plt
from baseline_lstm import LSTM_Model


with open('../trainlist.txt', 'r') as f:
    train_d = json.loads(f.read())


model = LSTM_Model(hidden_size=80)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=.001)

batch_size = 20
num_train_examples = len(train_d)
num_train_batches = int(num_train_examples/batch_size)

loss_function = torch.nn.BCELoss()

for batch_index in range(num_train_batches):
    batch_data = train_d[batch_index*batch_size:(batch_index+1)*batch_size]
    
    # TODO: convert batch_data to word vectors
    
    optimizer.zero_grad()
    predictions = model(batch_sentences)
    loss = loss_function(predictions)
    loss.backward()

    # clip gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # using 5.0 as default from HW4

    optimizer.step()