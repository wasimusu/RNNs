"""
RNNs to classify sentences as exclamation, question or statement.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchtext.datasets as datasets

from torchtext import data

TEXT = data.Field(tokenize='spacy', include_lengths=True)
LABEL = data.LabelField(dtype=torch.long)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
max_vocab_size = 25000

TEXT.build_vocab(train_data, vectors='glove.6B.300d',vectors_cache="../.vector_cache")
LABEL.build_vocab(train_data)
print("loaded pretrained vectors")

batch_size = 64

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

train_iterator,  test_iterator = data.BucketIterator.splits(
    (train_data, test_data), batch_size=batch_size, sort_within_batch=True, device=device
)

vocab_size = len(TEXT.vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = len(LABEL.vocab)


class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size, num_layers, num_directions, hidden_dim, output_dim, batch_size):
        super(SentenceClassifier, self).__init__()

        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
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
num_layers = 1
num_directions = 1

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = SentenceClassifier(vocab_size, num_layers, num_directions, hidden_dim, output_dim, batch_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(params=model.parameters(), lr=0.1)

embedding = torch.nn.Embedding(vocab_size, embedding_dim)
pretrained_embeddings = TEXT.vocab.vectors
model.embeddings.weight.data.copy_(pretrained_embeddings)

# Training parameters
num_epoch = 100

print("Started training")
for epoch in range(num_epoch):
    epoch_loss = 0
    for data in train_iterator:
        text, text_lengths = data.text
        output = model(text).squeeze(1).to(device)[-1].view(batch_size, -1)

        model.zero_grad()
        loss = criterion(output, data.label)
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss

    if epoch % 1 == 0:
        print("Epoch  {}/{} \tLoss : {}".format(epoch, num_epoch, "%.2f" % epoch_loss.item()))

torch.save(model.parameters(), "SentenceClassifier_model")
