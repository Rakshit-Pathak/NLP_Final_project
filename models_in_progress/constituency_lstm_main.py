import time
import pickle
import numpy as np
import torch
import torch.nn.utils
import json
from baseline_lstm import LSTM_Model

from datetime import datetime

import sys
sys.path.append('../') # needed to access functions in the parent directory

from constituency_preprocess import WVC

#with open('../trainlist.txt', 'r') as f:
#    data_train = json.loads(f.read())

with open('./const_preprocess_review.pkl', 'rb') as f:
    data_train = pickle.load(f)
    f.close()

with open('../testlist.txt', 'r') as f:
    data_test = json.loads(f.read())

with open('./dict_compressed.pickle', 'rb') as f:
    wv_dict = pickle.load(f)
    f.close()

###### preprocessing 
wvc = WVC(wv_dict)

model = LSTM_Model(hidden_size=80,input_dimension=316)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=.001)


#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch_train_accuracy_hist_loaded = checkpoint['training_history']
#epoch_test_accuracy_hist_loaded = checkpoint['test_history']

num_epochs = 10
batch_size = 10
print_every = 5 # prints running stats every X batches
num_train_examples = len(data_train)
num_train_batches = int(num_train_examples/batch_size)

development_set_fraction = 0.5
num_test_examples = int(development_set_fraction*len(data_test))
num_test_batches = int(num_test_examples/batch_size)

loss_function = torch.nn.BCELoss()

# split apart the text and labels
train_text,train_labels = zip(*data_train)
test_text,test_labels = zip(*data_test[:num_test_examples])

print_timing_statements = False

batch_loss_avg = 0
running_correct_avg = 0
total_epoch_correct_avg = 0
total_epoch_loss_avg = 0
running_correct_avg_hist = []
epoch_train_accuracy_hist = []
epoch_train_loss_hist = []
epoch_test_accuracy_hist = []
epoch_test_loss_hist = []
for epoch in range(num_epochs):
    for batch_index in range(num_train_batches):
        start_batch = time.time()
        start = time.time()
        
        batch_reviews_text = train_text[batch_index*batch_size:(batch_index+1)*batch_size]
        batch_labels = train_labels[batch_index*batch_size:(batch_index+1)*batch_size]
       
        end = time.time()
        if (print_timing_statements):
            print('----------------------------------------------')
            print('Batch assigment time: '+str(end-start))
        
        start = time.time()
        
        # embedding the text
        batch_reviews_vec = [wvc.const_preprocess(review) for review in batch_reviews_text]
        
        end = time.time()
        if (print_timing_statements):
            print('Batch word2vec time: '+str(end-start))
        
        start = time.time()
        
        # sort the batch sentences by length (will be necessary for pack_padded_sequence)
        review_lens = [-len(review) for review in batch_reviews_vec]
        sort_inds = np.argsort(review_lens)
        batch_reviews_vec_sorted = [batch_reviews_vec[i] for i in sort_inds]
        batch_labels_sorted = [batch_labels[i] for i in sort_inds]
        
        end = time.time()
        if (print_timing_statements):
            print('Sorting time: '+str(end-start))
        
        start = time.time()
        
        optimizer.zero_grad()
        predictions = model(batch_reviews_vec_sorted)
    
        end = time.time()
        if (print_timing_statements):
            print('Predictions time: '+str(end-start))
    
        start = time.time()
       
        loss = loss_function(torch.squeeze(predictions),torch.tensor(batch_labels_sorted, dtype=torch.float))
        loss.backward()
    
        end = time.time()
        if (print_timing_statements):
            print('Backprop time: '+str(end-start))
    
        start = time.time()
    
        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # using 5.0 as default from HW4
    
        optimizer.step()
    
        end = time.time()
        if (print_timing_statements):
            print('Optimizer step time: '+str(end-start))
            print('----------------------------------------------')
        
        batch_loss = loss.sum()
        batch_losses_val = batch_loss.item()
        batch_loss_avg += batch_losses_val/print_every
        batch_correct = np.sum(np.equal(np.greater(np.squeeze(predictions.detach().numpy()),0.5),np.asarray(batch_labels_sorted)))/batch_size
        running_correct_avg += batch_correct/print_every
        total_epoch_correct_avg += batch_correct
        total_epoch_loss_avg += batch_loss_avg
        end_batch = time.time()
        
        if batch_index % print_every == 0 and batch_index > 0:
            print('Batch '+str(batch_index)+'/'+str(num_train_batches)+', loss: '+str(batch_loss_avg)+', running accuracy: '+str(running_correct_avg),', time/batch: '+str(end_batch-start_batch))
            running_correct_avg_hist.append(running_correct_avg)
            batch_loss_avg = 0
            running_correct_avg = 0
            
    epoch_accuracy = total_epoch_correct_avg/num_train_batches
    total_epoch_loss_avg = total_epoch_loss_avg/num_train_batches
    print('----------------------------------------------')
    print('----------------------------------------------')
    print('END OF EPOCH '+str(epoch+1)+', TOTAL EPOCH TRAINING ACCURACY = '+str(epoch_accuracy)+', AVG EPOCH TRAINING LOSS = '+str(total_epoch_loss_avg))
    epoch_train_accuracy_hist.append(epoch_accuracy)
    epoch_train_loss_hist.append(total_epoch_loss_avg)
    total_epoch_correct_avg = 0
    total_epoch_loss_avg = 0
    
    # at end of epoch do a test loop 
    total_test_correct_avg = 0
    total_test_loss_avg = 0
    print('running test on development set')
    for test_batch_index in range(num_test_batches):
    
        batch_reviews_text = test_text[test_batch_index*batch_size:(test_batch_index+1)*batch_size]
        batch_labels = test_labels[test_batch_index*batch_size:(test_batch_index+1)*batch_size]
       
        # embedding the text
        batch_reviews_vec = [wvc.const_preprocess(review) for review in batch_reviews_text]

        # sort the batch sentences by length (will be necessary for pack_padded_sequence)
        review_lens = [-len(review) for review in batch_reviews_vec]
        sort_inds = np.argsort(review_lens)
        batch_reviews_vec_sorted = [batch_reviews_vec[i] for i in sort_inds]
        batch_labels_sorted = [batch_labels[i] for i in sort_inds]
        
        predictions = model(batch_reviews_vec_sorted)
    
        batch_loss = loss.sum()
        batch_losses_val = batch_loss.item()
        total_test_loss_avg += batch_losses_val/batch_size
        batch_correct = np.sum(np.equal(np.greater(np.squeeze(predictions.detach().numpy()),0.5),np.asarray(batch_labels_sorted)))/batch_size
        total_test_correct_avg += batch_correct
    
    total_test_correct_avg = total_test_correct_avg/num_test_batches
    total_test_loss_avg = total_test_loss_avg/num_test_batches
    epoch_test_accuracy_hist.append(total_test_correct_avg)
    epoch_test_loss_hist.append(total_test_loss_avg)
    print('EPOCH '+str(epoch+1)+', TOTAL EPOCH TEST ACCURACY = '+str(total_test_correct_avg)+' AVG TEST LOSS: '+str(total_test_loss_avg))
    print('----------------------------------------------')
    print('----------------------------------------------')
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_history': epoch_train_accuracy_hist,
                'test_history': epoch_test_accuracy_hist
                }, './const_parsing_'+str(epoch)+'epoch_80hidden'+now.strftime("%m%d%Y_%H_%M_%S"))
    
