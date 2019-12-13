# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:46:56 2019

@author: Peter
"""

import json
import numpy as np
from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# this is modified from HW4 -- instead it does padding after the embedding so it works on lists (length batch_size) of numpy arrays of shape [T,embedded_dim] 
#def pad_sents(sents, pad_token):
#
#    sents_padded = []
#    max_len = max(np.shape(s)[0] for s in sents)
#    for s in sents:
#        padded = np.zeros([max_len,300])
#        padded[:np.shape(s)[0]] = s
#        padded[np.shape(s)[0]:] = pad_token
#        sents_padded.append(padded)
#    return sents_padded

# this is from HW4
def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []
    max_len = max(len(s) for s in sents)
    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)
    return sents_padded

# this is from HW4
def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

class LSTM_Model(nn.Module):
    
    def __init__(self, hidden_size):
        super(LSTM_Model, self).__init__()


        self.hidden_size = hidden_size

        self.LSTM = nn.LSTM(300, hidden_size, bidirectional=True)
        
        # Classification Layer
        self.out = nn.Linear(2*hidden_size, 1) # 2*hidden_size because it will be using the hidden state at t=0 from the backward LSTM and t=T from the forward LSTM
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, sequences):
        
        # sequences is of size [B,T,embedded_dim]
        batch_size = len(sequences)
        
        # Using pack_padded_sequence
        source_lengths = [len(s) for s in sequences]
        sequences_padded = pad_sents(sequences,np.zeros([300],dtype='float32')) # assuming padding token is 0
        sequences_padded = torch.tensor(sequences_padded, dtype=torch.float)
        sequences_packed_padded = torch.nn.utils.rnn.pack_padded_sequence(torch.transpose(sequences_padded,0,1),source_lengths)
        
        # LSTM returns: outputs, (hidden_states, cell_states)
        # outputs are at all timesteps, (hidden_states, cell_states) are at final timesteps
        outputs, (hidden_states, cell_states) = self.LSTM(sequences_packed_padded)
        
        # want to use the final hidden state of the forward LSTM and first state of backward LSTM
        lstm_out = torch.cat((hidden_states[0],hidden_states[1]),dim = 1)
        out = self.sigmoid(self.out(lstm_out))
        
        return out
    
