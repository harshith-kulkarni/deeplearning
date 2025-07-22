# Regrression Task Implementation using Artificial Neural Network
In this repo i have implemented regression task by building an ANN in order to predict the home price 
dataset used : https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv

Built a model from scratch using numpy and sklearn

**Steps to follow**
1. import modules required
2. import dataset
3. standardise / normalise the data if necessary
4. get rid of outliers 
5. initialize parameter sizes (input , no of hidden layers , no of neurons in hidden layer)
6. initialise weights and biases
7. define losses , metrics
8. define forward propagation
9. define back propagation
10. define update parameters
11. training
12. testing
13. visualtisation of losses
14. evaluation

**model specifications**
standardised the data
to update parameters use gradient descent algorithm 
learning_rate=0.01 
epochs=1000
evaluation metric : MSE (mean squared error)
activation function : RELU , RELU_derivative for backward propagation

**RESULTS**
final loss : epoch 990, cost=0.9994459071828705