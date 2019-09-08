import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import ipdb
from fourier import FourierVectorizer

class BiLSTM(nn.Module):

    def __init__(self,
            embeddings,
            hidden_size,
            num_labels,
            bidirectional=False, 
            num_layers=1,
            color_representation_size=54):

        super(BiLSTM, self).__init__()
        self.embeddings = embeddings
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.input_size = self.embeddings.embedding_dim
        self.color_representation_size= color_representation_size

        self.lstm = nn.LSTM(self.input_size + self.color_representation_size,
                            hidden_size,
                            bidirectional=bidirectional,
                            num_layers=num_layers,
                            batch_first=False,
                            dropout= 0.2)

        self.total_hidden_size = \
                self.hidden_size * 2 if self.bidirectional else self.hidden_size


        self.output_layer = nn.Linear(self.total_hidden_size, self.num_labels)

        self.loss_function = nn.CrossEntropyLoss()
        # https://pytorch.org/docs/0.3.1/nn.html#torch.nn.CrossEntropyLoss
        self.is_cuda = False

    def cuda(self, *args, **kwargs):
        super(BiLSTM, self).cuda(*args, **kwargs)
        self.is_cuda = True

    def zero_state(self, batch_size=1):
        h_0 = Variable(torch.zeros(self.num_layers, 
                                    batch_size, 
                                    self.hidden_size))

        c_0 = Variable(torch.zeros(self.num_layers,
                                    batch_size,
                                    self.hidden_size))

        return (h_0, c_0)

    def forward(self, sequence_batch, sequence_lengths,
            color_vectors=None, train_embeddings=False):

        seq_len, batch_size = sequence_batch.size()
        embedded_sequence_batch = self.embeddings(sequence_batch)

        tgt_batch = embedded_sequence_batch[0, :].unsqueeze(0)
    
        cv = FourierVectorizer([3], hsv=True) 
        color_vectors_ = cv.vectorize_all(color_vectors.data.cpu().numpy()[0])
        color_vectors_ = torch.FloatTensor(color_vectors_)
        color_vectors_ = Variable(color_vectors_)
        color_vectors_ = color_vectors_.cuda()
        color_vectors = color_vectors_.unsqueeze(0)

        combined_vectors = torch.cat((color_vectors, tgt_batch), 2)
        
        logits = [] 
        predictions = []
        labels = sequence_batch[1:, :]

        hidden = self.zero_state(tgt_batch.size(1))
        hidden = (hidden[0].cuda(), hidden[1].cuda())

        #ipdb.set_trace()

        for i in range(seq_len - 1):
             
            out, hidden = self.lstm(combined_vectors, hidden)     
            logits_i = self.output_layer(out)
            logits.append(logits_i)
    
            _, predictions_i = logits_i.max(2)

            predictions.append(predictions_i)
            predictions_i_embedded = self.embeddings(predictions_i)
            combined_vectors = torch.cat((color_vectors, predictions_i_embedded), 2)

        flat_logits_ = torch.cat(logits, 0)
        flat_logits = flat_logits_.view( batch_size * (seq_len -1), self.num_labels)
        
        flat_labels = labels.view(-1)

        loss = self.loss_function(flat_logits, flat_labels)

        return loss, predictions
