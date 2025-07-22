# Binary Classification Task Implementation using Artificial Neural Network
In this repo i have implemented classification task by building an ANN in order to predict the whether a patient has breast-cancer or not
dataset used :IRIS
Built a model from scratch using numpy and sklearn

Features:
Target: Iris-setosa Iris-versicolor Iris-virginica

**Model Architecture**
A simple 3-layer Neural Network:

Input Layer: 5 features

Hidden Layer: 10 neurons with ReLU activation

Output Layer: 3 neuron with Softmax activation (for multi class classification)

Loss Function: Categorical Cross Entropy
Optimizer: Manual Gradient Descent

**Training Flow**
Data Preprocessing

Loaded and cleaned the dataset

Normalized the feature values

Converted labels : 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2

Model Training

Forward Propagation

Loss Calculation (Categorical Cross Entropy)

Backpropagation for parameter updates

**model specifications**
standardised the data
to update parameters use gradient descent algorithm 
learning_rate=0.05
epochs=500
evaluation metric : Accuracy score
activation function : sigmoid , sigmoid_derivative for backward propagation , Softmax for output 
Added evaluation method to take inputs from user and make predictions

**Result** 
Final accuracy : 98 %