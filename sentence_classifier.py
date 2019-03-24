"""
RNNs to train parts of speech tagger
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from nltk import WordPunctTokenizer

wordTokenizer = WordPunctTokenizer()

import os


class DataIteratior:
    def __init__(self, filename, batch_size, MAX_LEN=20):
        if not os.path.exists(filename):
            raise ValueError(filename, ' does not exist. ')

        string = open(filename).read().lower()
        vocab = set(wordTokenizer.tokenize(string))
        vocab_size = len(vocab)
        print("Vocab size : ", vocab_size)

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_len = MAX_LEN

        self.word_to_index = dict(zip(vocab, list(range(len(vocab)))))
        self.index_to_word = dict(zip(list(range(len(vocab))), vocab))
        self.label_to_id = {"?": 0, ".": 1, "!": 2}

        sentences = string.splitlines()
        self.sentences = [wordTokenizer.tokenize(sentence) for sentence in sentences if len(sentence) > 5]
        self.sentences = [[sentence[:-1], sentence[-1]] for sentence in self.sentences if
                          sentence[-1] in self.label_to_id.keys()]

        # Split columns
        self.sentences, self.targets = zip(*self.sentences)
        # Because tuple data type can not be popped
        self.sentences = list(self.sentences)
        self.targets = list(self.targets)

        self.num_batches = len(sentences) // batch_size
        self.sentences = self.sentences[:self.num_batches * batch_size]

    def __next__(self):
        sentences = [self.sentences.pop() for _ in range(self.batch_size)]
        inputs = []
        for sentence in sentences:
            inputs.append([self.word_to_index[word] for word in sentence])
        labels = [self.label_to_id[self.targets.pop()] for _ in range(self.batch_size)]
        return inputs, labels

    def __len__(self):
        return self.num_batches


class Tagger(nn.Module):
    def __init__(self, vocab_size, num_layers, num_directions, hidden_dim, output_dim, batch_size):
        super(Tagger, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True if self.num_directions != 1 else False)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        self.hidden = self.initHidden()

    def forward(self, x):
        """ Take x in degrees """
        x = self.embeddings(x)
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.relu(x)
        x = self.linear(x)
        x = F.softmax(x)
        return x

    def initHidden(self):
        return (torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to(device))


# Parameters of the model
batch_size = 1
data = DataIteratior("data/lines.txt", batch_size, 20)
vocab_size = data.vocab_size
num_layers = 1
num_directions = 1
hidden_dim = 80
output_dim = 3

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = Tagger(vocab_size, num_layers, num_directions, hidden_dim, output_dim, batch_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(params=model.parameters(), lr=0.1)

# Training parameters
num_epoch = 100

print("Started training")
for epoch in range(num_epoch):
    epoch_loss = 0
    for _ in range(len(data)):
        inputs, labels = data.__next__()
        inputs = torch.tensor(inputs).reshape(-1, 1).long().to(device)
        labels = torch.tensor(labels).long().to(device)

        output = model(inputs).squeeze(1).to(device)[-1].view(batch_size, -1)

        model.zero_grad()
        loss = criterion(output, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss

    if epoch % 2 == 0:
        print("Epoch  {}/{} \tLoss : {}".format(epoch, num_epoch, "%.2f" % epoch_loss.item()))

torch.save(model.parameters(), "tagger_model")
