import numpy as np
import pandas as pd
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download dataset from Kaggle
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
print("Path to dataset files:", path)

# Load dataset
df = pd.read_csv(path + "/data.csv")
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
df = df.drop(['id', 'Unnamed: 32'], axis=1)  # Drop unnecessary columns
print(df.head())
# Plot histogram
df.hist(bins=30, figsize=(20, 10))
plt.show()

# Prepare features and labels
x = df.drop(["diagnosis"], axis=1).values
y = df["diagnosis"].values.reshape(-1, 1)

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Model parameters
input_size = x_train.shape[1]
hidden_size = 10
output_size = 1

# Initialize weights
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Forward propagation
def forward_propagation(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Binary cross-entropy loss
def binary_cross_entropy(y, a2):
    epsilon = 1e-15
    a2 = np.clip(a2, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))

# Backward propagation
def backward_propagation(x, y, z1, a1, a2, w2):
    m = x.shape[0]
    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * sigmoid_derivative(a1)
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    return dw1, db1, dw2, db2

# Update weights
def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2

# Training function
def train_model(x, y, hidden_size, learning_rate, epochs):
    input_size = x.shape[1]
    output_size = 1
    w1, b1, w2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    accuracy_list = []

    for i in range(epochs):
        z1, a1, z2, a2 = forward_propagation(x, w1, b1, w2, b2)
        loss = binary_cross_entropy(y, a2)
        y_pred = (a2 > 0.5).astype(int)
        accuracy = np.mean(y_pred == y)
        accuracy_list.append(accuracy)

        dw1, db1, dw2, db2 = backward_propagation(x, y, z1, a1, a2, w2)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return w1, b1, w2, b2, accuracy_list

# Train the model
w1, b1, w2, b2, acc_list = train_model(x_train, y_train, hidden_size=10, learning_rate=0.05, epochs=1000)

# Plot training accuracy
plt.plot(acc_list)
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Over Time")
plt.show()

# Evaluate on test set
def predict(x, w1, b1, w2, b2):
    _, _, _, a2 = forward_propagation(x, w1, b1, w2, b2)
    return (a2 > 0.5).astype(int)

y_pred_test = predict(x_test, w1, b1, w2, b2)
test_accuracy = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
