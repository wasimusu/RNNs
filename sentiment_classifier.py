"""
RNNs to classify sentences as exclamation, question or statement.
"""

import math
import random
import os
import urllib.request as request

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

embedding_dim = 100
hidden_dim = embedding_dim
device = ('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# Parameters of the model
num_layers = 1
num_directions = 1
num_epoch = 100


class IMDB:
    def __init__(self, max_vocab_size):
        self.urls = ['http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz']
        self.data_root = "data"
        self.name = 'imdb'
        self.maybe_download()

    def maybe_download(self):
        data = os.path.join(self.data_root, self.name)
        if not os.path.exists(data):
            os.makedirs(data)
        request.urlretrieve(url=self.urls, filename=self.name)

    def tokenize(sequence):
        tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
        return tokens

    def tokenize(self):
        pass

    def process(self):
        pass

    def __next__(self):
        pass

    def batch_size(self):
        return

    def next(self):
        return self.__next__()


class SentenceClassifier(nn.Module):

    def get_pretrained_embedding(self, np_emb_matrix):
        embeddings = nn.Embedding(*np_emb_matrix.shape)
        embeddings.weight = nn.Parameter(torch.from_numpy(np_emb_matrix).float())
        embeddings.weight.requires_grad = False
        return embeddings

    def __init__(self, embedding, num_layers, hidden_dim, output_dim, batch_size, bidirectional=False):
        super(SentenceClassifier, self).__init__()

        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=bidirectional,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.embeddings = self.get_pretrained_embedding(embedding)

        self.hidden = self.initHidden()

    def forward(self, inputs, input_lengths):
        """ Take x in degrees """

        # Sort the sequence lengths
        lens_sorted, lens_argsort = torch.sort(input_lengths, dim=0, descending=True)

        # Sorting the argsort sorts the original indices
        # which of course is equivalent to putting the original indices
        # in the original order, thus, argsort_argsort are the positions used
        # to restore the sorted version back to the original.
        _, lens_argsort_argsort = torch.sort(lens_argsort, dim=0)

        # Convert the numbers into embeddings
        inputs = self.embeddings(inputs.to('cpu'))
        # packed = inputs

        # Get the sorted version of inputs as required for pack_padded_sequence
        inputs_sorted = torch.index_select(inputs, 0, lens_argsort)

        packed = pack_padded_sequence(inputs_sorted, lens_sorted, batch_first=True)
        output, self.hidden = self.encoder(packed, self.hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        x = self.embeddings(x)
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.relu(x)
        x = self.linear(x)
        x = F.softmax(x)
        return x

    def initHidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to(device)


def train():
    # Model, loss and optimizer
    model = SentenceClassifier(embedding, num_layers, hidden_dim, output_dim, batch_size, bidirectional=False).to(
        device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(params=model.parameters(), lr=0.1)

    print("Started training")
    for epoch in range(num_epoch):
        epoch_loss = 0
        for data in train_iterator:
            text, text_lengths = data.text
            output = model(text, text_lengths)

            model.zero_grad()
            loss = criterion(output, data.label)
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss

        if epoch % 1 == 0:
            print("Epoch  {}/{} \tLoss : {}".format(epoch, num_epoch, "%.2f" % epoch_loss.item()))

    torch.save(model.parameters(), "SentenceClassifier_model")


if __name__ == '__main__':
    pass
