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
def pad_sents(sents, pad_token):

    sents_padded = []
    max_len = max(np.shape(s)[0] for s in sents)
    for s in sents:
        padded = np.zeros([max_len,300])
        padded[:np.shape(s)[0]] = s
        padded[np.shape(s)[0]:] = pad_token
        sents_padded.append(padded)
    return sents_padded


class LSTM_Model(nn.Module):
    
    def __init__(self, hidden_size):
        super(LSTM_Model, self).__init__()


        self.hidden_size = hidden_size

        self.LSTM = nn.LSTM(300, hidden_size, bidirectional=True)
        
        # Classification Layer
        self.out = nn.Linear(hidden_size, 1)
    
    def forward(self, sequences):
        
        # sequences is of size [T,B,embedded_dim]
        batch_size = sequences.size(1)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        
        # Using pack_padded_sequence
        source_lengths = [len(s) for s in sequences]
        sequences_padded = pad_sents(sequences,0) # assuming padding token is 0
        sequences_padded = torch.tensor(sequences_padded, dtype=torch.long)
        
        sequences_packed_padded = torch.nn.utils.rnn.pack_padded_sequence(sequences_padded,source_lengths)
        
        # LSTM returns: outputs, (hidden_states, cell_states)
        # outputs are at all timesteps, (hidden_states, cell_states) are at final timesteps
        outputs, (hidden_states, cell_states) = self.LSTM(sequences_packed_padded, source_lengths)
        
        # want to use the final hidden state of the forward LSTM and first state of backward LSTM
        lstm_out = torch.cat((hidden_states[0],hidden_states[1]),dim = 1)
        
        out = torch.nn.Sigmoid(self.out(lstm_out))
        
        return out, hidden
    
