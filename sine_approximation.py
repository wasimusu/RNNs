"Using LSTM to approximate sine function. Does not work"

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Sine(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, num_layers=1, bidiectional=False):
        super(Sine, self).__init__()

        # RNN Parameters
        self.num_layers = num_layers
        self.num_directions = 2 if bidiectional else 1
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True if self.num_directions != 1 else False)
        self.linear = nn.Linear(hidden_dim * self.num_directions, output_dim)

        self.hidden = self.initHidden()

    def forward(self, x):
        """ Take x in degrees """
        x, self.hidden = self.lstm(x, self.hidden)

        # There are many activation functions to try. F.tanh | F.leaky_relu | F.relu
        x = torch.tanh(x)
        x = self.linear(x)
        return x

    def initHidden(self):
        return (torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim))


# Parameters of the model
input_dim = 1
num_layers = 1
bidiectional = False
hidden_dim = 256 * 2
output_dim = 1
batch_size = 8

# Defining model, optimizer and loss function
model = Sine(input_dim,
             hidden_dim,
             output_dim,
             batch_size,
             num_layers=1,
             bidiectional=False)

optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.00, weight_decay=0.0001)
criterion = nn.MSELoss()

# Training parameters
num_epoch = 100

count = 0
for epoch in range(num_epoch):
    epoch_loss = 0
    for i in range(10):
        input = [random.uniform(count, count + 1) for _ in range(batch_size)]
        label = [math.sin(x) for x in input]  # The inputs are in radians
        input = torch.tensor([input]).reshape(-1, 1).float().unsqueeze(0)
        label = torch.tensor(label)
        output = model(input)

        model.zero_grad()
        loss = criterion(output, label)
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss
        count += 1

    if epoch % 10 == 0:
        print("Epoch  {}\tLoss : {}".format(epoch, "%.2f" % epoch_loss.item()))

    if epoch % 20 == 0 and epoch >= 20:
        with torch.no_grad():
            # Test the model
            input = [0, 45, 90, 135, 180, 225, 270, 315]
            label = [math.sin(math.radians(x)) for x in input]
            inputs = torch.tensor([input]).reshape(-1, 1).float().unsqueeze(0)
            output = model(inputs)
            print("Input : ", input, "\nOutput : ", np.round(output.view(1, -1).detach().numpy(), 2),
                  "\nLabel : ", np.round(label, 2), '\n')

    if count == 100:
        count = 0
