# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:46:56 2019

@author: Peter
"""

import json

from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

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
    
