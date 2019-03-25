import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def generate_data(N, sigma):
    """ Generate data with given number of points N and sigma """
    noise = np.random.normal(0, sigma, N)
    X = np.random.uniform(0, 3, N)

    # More work required to make it work on this. Works decently.
    Y = 2 * X ** 2 + 3 * X + 1 + noise  # Compute y from x

    # Works better on this
    # Y = X * 2 + 1 + noise  # Compute y from x

    return X, Y


class Regression(nn.Module):
    def __init__(self, input_dim, num_layers, num_directions, hidden_dim, output_dim, batch_size):
        super(Regression, self).__init__()

        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

        self.hidden = self.initHidden()

    def forward(self, x):
        """ Take x in degrees """
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.relu(x)
        x = self.linear(x)
        return x

    def initHidden(self):
        return (torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim))


# Parameters of the model
input_dim = 1
num_layers = 1
num_directions = 1
hidden_dim = 80
output_dim = 1
batch_size = 8

model = Regression(input_dim, num_layers, num_directions, hidden_dim, output_dim, batch_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.1)

# Training parameters
num_epoch = 1000

for epoch in range(num_epoch):
    inputs, labels = generate_data(N=batch_size, sigma=0)
    inputs = torch.tensor(inputs).reshape(-1, 1).float().unsqueeze(0)
    labels = torch.tensor(labels).reshape(-1, 1).float()

    output = model(inputs)

    model.zero_grad()
    loss = criterion(output, labels)
    loss.backward(retain_graph=True)
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch  {}\tLoss : {}".format(epoch, "%.2f" % loss.item()))

        # Test the model
        input, labels = generate_data(N=batch_size, sigma=0)
        inputs = torch.tensor(input).reshape(-1, 1).float().unsqueeze(0)

        with torch.no_grad():
            outputs = model(inputs)
            inputs = torch.tensor([input]).reshape(-1, 1).float().unsqueeze(0)
            outputs = np.round(outputs.view(1, -1).detach().numpy(), 2)
            print(np.round(input, 2), '\n',
                  np.round(labels, 2), '\n', outputs, '\n\n')
