import torch
import torch.nn as nn
import tqdm
import os
import random
from data import read_train_data, read_test_data
from utils import Vocab, BatchIterator
from model import BiLSTM
import ipdb 
from datetime import datetime
from tensorboardX import SummaryWriter
import time

train_path = "/home/user/data/rugstk/data/munroecorpus/train"
test_path  = "/home/user/data/rugstk/data/munroecorpus/25test"
output_path = "/home/user/data/color-description/"

epochs = 20
batch_size = 128 
hidden_size = 100
log_interval = 500
bidirectional = False
num_layers = 1
min_count = 1
gradient_clipping = 0.2 
embedding_size = 100
cuda = True

print(str(datetime.now()), "Loading data")
train_colors = read_train_data(train_path)
test_colors  = read_test_data(test_path)

random.shuffle(train_colors)
random.shuffle(test_colors)

print(str(datetime.now()), "Generating vocab")
vocab = Vocab(train_colors, 
        min_count=min_count, 
        add_padding=True,
        add_bos=True,
        add_eos=True) 

embeddings = nn.Embedding(len(vocab.index2token),
                            embedding_size,
                            padding_idx= vocab.PAD.hash)

model = BiLSTM(embeddings=embeddings,
        hidden_size=hidden_size,
        num_labels= len(vocab),  #num_labels,
        bidirectional=bidirectional,
        num_layers=num_layers,
        color_representation_size=54)#54)


model_id = str(int(time.time())) + "w_fourier" 
save_path = os.path.join(output_path, model_id)
if not os.path.isdir(save_path):
    os.makedirs(save_path)

writer = SummaryWriter(save_path)


if cuda:
    model.cuda()

print(model)

print(str(datetime.now()), 'Generating batches')
train_batches = BatchIterator(train_colors, vocab, batch_size, cuda=cuda)
test_batches = BatchIterator(test_colors, vocab, batch_size, cuda=cuda)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.5)

pbar = tqdm.trange(epochs, desc='Training...')

delta = 0 
delta_test = 0 

for epoch in pbar:
    total_loss = 0
    total_batches = 0
  
    for i, batch in enumerate(train_batches):
        (id_slice, padded_y_slice, y_slice_lengths, x_slice) = batch

        loss, predictions  = model.forward(padded_y_slice,
                        y_slice_lengths,
                        x_slice)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clipping)
        optimizer.step()
        
        total_batches +=1
        total_loss += loss.data[0]
        

        if i % log_interval == 0 and i > 0:
            writer.add_scalar('train_loss_per_batch', total_loss/ total_batches, i + delta)
            total_loss = 0 
            total_batches = 0 
        
    delta = delta + len(train_batches)

    # testing
    
    comparison = [] 

    total_loss_test = 0
    total_batches_test = 0 

    for i, batch in enumerate(test_batches):
        (id_slice, padded_y_slice, y_slice_lengths, x_slice) = batch
        test_loss, test_predictions = model.forward(padded_y_slice,
                y_slice_lengths,
                x_slice)


        test_predictions = torch.cat(test_predictions,0)
        test_predictions = test_predictions.transpose(0,1)
        references = padded_y_slice.transpose(0,1)
        color_vectors = x_slice.data.cpu().numpy()[0] # a bit ugly

        for test_prediction, reference, color in zip(test_predictions, references, color_vectors):
            comparison.append((
                    [v.string for v in vocab.indices2tokens(test_prediction.data)],
                    [v.string for v in vocab.indices2tokens(reference.data)],
                    [str(ck) for ck in  color]

            ))
        

        total_batches_test +=1
        total_loss_test += test_loss.data[0]

        if i % log_interval == 0  and i > 0 :
            writer.add_scalar('test_loss_per_batch', total_loss_test/ total_batches_test, i + delta_test)
            total_loss_test = 0 
            total_batches_test = 0 
    
    delta_test = delta_test + len(test_batches)
    with open(output_path+str(epoch)+".txt", 'w') as f:
        for (p, r, c) in comparison:
            f.write(" ".join(p) +"|" + " ".join(r) +"|" + " ".join(c) +  "\n")

