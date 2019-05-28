## Tutorial on Recurrent Neural Networks in PyTorch
Here we cover the following topics
- Saving and restoring trained models
- L1 and L2 regularization
- Using different flavors of RNNs like LSTM, GRU
- Using RNN for different usage like regression and MNIST handwritten digit classification
- The programs are self contained for ease of understanding
- Using dropout in images classification

### Numerical Regression using LSTM
- Setting up bidirectional and multilayer RNNs.
- Testing out different activation functions because numerical regression is different from other tasks like
classification and thus demands a bit different activation function
- L2 regularization
- filename : [linear_regression.py](https://github.com/wasimusu/RNNs/blob/master/linear_regression.py)

### MNIST Handwritten digit classifier using LSTM
- L2 regularization
- Using dropout in image classification
- Saving and restoring models
- Using MNIST images from torchvision
- Moving models to specific device (GPU / CPU)
- Setting up bidirectional and multilayer RNNs.
- filename : [mnist_classifier.py](https://github.com/wasimusu/RNNs/blob/master/mnist_classifier.py)

### MNIST Handwritten digit classifier using GRU
- Same as above but uses Gated Recurrent Unit (GRU)
- filename : [mnist_classifier.py](https://github.com/wasimusu/RNNs/blob/master/mnist_classifier_gru.py)

### Sine Approximation using LSTM - Does not work (yet)
- Learning to use different activation functions
- filename : [sine_approximation.py](https://github.com/wasimusu/RNNs/blob/master/sine_approximation.py)
