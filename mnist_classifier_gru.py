"""
MNIST being trained by using RNNs
Concepts used :
 - Using pretrained networks
 - Dropout
 - l1 regularization
 - l2 regularization
 - Hyper-parameter tuning
 - Saving and restoring model
 - Moving models to gpu or cpu
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

learning_rate = 0.05  # Softmax requires especially low learning rates
momentum = 0.1  # Momentum and dropout help to speed up training but they're not always helpful
reuse_model = True
delta_loss = 0.01  # The minimum threshold differences required between two consecutive epoch to continue training
l2 = 0.99

device = ('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_filename = "model/rnn_mnist_model"


def getAccuracy(dataLoader):
    """ Compute accuracy for given dataset """
    total, correct = 0, 0
    for i, data in enumerate(dataLoader):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            if inputs.size(0) != batch_size: continue

            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1)
            score = sum(outputs == labels).data.to('cpu').numpy()

            total += batch_size
            correct += score

    accuracy = correct * 1.0 / total
    return accuracy


class MnistNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, bidirectional=False, dropout=0.0):
        super(MnistNet, self).__init__()

        # Parameters of the LSTM Network
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.features = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

        self.hidden = self.initHidden()

    def forward(self, x):
        x = x.view(1, batch_size, -1)
        x, self.hidden = self.features(x, self.hidden)

        x = x.view(-1, self.hidden_dim)  # Resize the input so that it can be fed to classifier / LinearLayer
        x = self.dropout(x)  # Apply dropout
        x = self.classifier(x)
        x = F.softmax(x, dim=0)
        return x

    def initHidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim).to(device)


# Parameters of the model
# You can change any of the parameters and expect the network to run without error
input_dim = 28 * 28
hidden_dim = 200
num_layers = 1
bidirectional = False
num_classes = 10
batch_size = 32
dropout = 0.01

# Instantiating a network to train on MNIST
model = MnistNet(input_dim,
                 hidden_dim,
                 num_classes,
                 num_layers=num_layers,
                 bidirectional=bidirectional,
                 dropout=0.0)

# Defining optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

# Use pretrained model or train new
if reuse_model == True:
    if os.path.exists(checkpoint_filename):
        model.load_state_dict(torch.load(f=checkpoint_filename))
    else:
        print("No pre-trained model detected. Starting fresh model training.")
model.to(device)

# Convert the image to tensor and normalize it
transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))]
)

# Dataset for training, validation and test set
trainset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=False, train=True)
trainset, validationset = torch.utils.data.random_split(trainset, [50000, 10000])
testset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=False, train=False)

num_train_samples = trainset.__len__()

# Data loader for train, test and validation set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)
validationloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)

# Train the model and periodically compute loss and accuracy on test set
epoch = 1
cur_epoch_loss = 10
prev_epoch_loss = 20
while abs(prev_epoch_loss - cur_epoch_loss) >= delta_loss:
    epoch_loss = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data

        # Putting the inputs onto GPU / CPU and squeezing to get the desired shape
        inputs = inputs.to(device).squeeze(1)
        labels = labels.to(device)

        if inputs.size(0) != batch_size: continue

        output = model(inputs)

        model.zero_grad()
        loss = criterion(output, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss

    print("{} Epoch. Loss : {}".format(epoch, "%.3f" % epoch_loss))

    # Every ten epochs compute validation accuracy
    if epoch % 10 == 0:
        print("{} Epoch. Accuracy on validation set : {}".format(epoch, "%.3f" % getAccuracy(validationloader)))

    # Save the model every ten epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f=checkpoint_filename)
        print()

    epoch += 1  # Incremenet the epoch counter
    prev_epoch_loss = cur_epoch_loss
    cur_epoch_loss = epoch_loss

# Do inference on test set
print("Accuracy on test set : {}".format("%.4f" % getAccuracy(testloader)))
