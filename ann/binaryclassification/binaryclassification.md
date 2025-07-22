# Binary Classification Task Implementation using Artificial Neural Network
In this repo i have implemented classification task by building an ANN in order to predict the whether a patient has breast-cancer or not
dataset used : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?select=data.csv
Built a model from scratch using numpy and sklearn

Features: 30 numeric features computed from digitized images of a breast mass

Target: diagnosis (M = Malignant, B = Benign)

**Model Architecture**
A simple 3-layer Neural Network:

Input Layer: 30 features

Hidden Layer: 16 neurons with ReLU activation

Output Layer: 1 neuron with Sigmoid activation (for binary classification)

Loss Function: Binary Cross Entropy
Optimizer: Manual Gradient Descent

**Training Flow**
Data Preprocessing

Loaded and cleaned the dataset

Normalized the feature values

Converted labels (M and B) into binary (1 and 0)

Model Training

Forward Propagation

Loss Calculation (Binary Cross Entropy)

Backpropagation for parameter updates

**model specifications**
standardised the data
to update parameters use gradient descent algorithm 
learning_rate=0.05
epochs=1000
evaluation metric : Accuracy score
activation function : sigmoid , sigmoid_derivative for backward propagation