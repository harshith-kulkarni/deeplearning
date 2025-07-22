import numpy as np
import pandas as pd
import kagglehub
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download and load dataset
path = kagglehub.dataset_download("uciml/iris")
df = pd.read_csv(path + "/Iris.csv")
print(df.head())

# Map species to integers
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Plot histogram
df.hist(bins=30, figsize=(20, 10))
plt.show()

# Features and labels
x = df.drop(["Species","Id"], axis=1).values
y = df["Species"].values.reshape(-1, 1)

# One-hot encode labels and convert to dense array
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y).toarray() 

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Parameters
input_size = x_train.shape[1]
hidden_size = 10
output_size = 3  # 3 classes

# Initialize weights
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2

# Activation functions
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Loss function
def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

# Forward propagation
def forward_propagation(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# Backward propagation
def backward_propagation(x, y, z1, a1, a2, w2):
    m = x.shape[0]
    dz2 = a2 - y 
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
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
    output_size = y.shape[1]
    w1, b1, w2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    accuracy_list = []

    for i in range(epochs):
        z1, a1, z2, a2 = forward_propagation(x, w1, b1, w2, b2)
        loss = categorical_cross_entropy(y, a2)

        y_pred_labels = np.argmax(a2, axis=1)
        y_true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred_labels == y_true_labels)
        accuracy_list.append(accuracy)

        dw1, db1, dw2, db2 = backward_propagation(x, y, z1, a1, a2, w2)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return w1, b1, w2, b2, accuracy_list

# Train the model
w1, b1, w2, b2, acc_list = train_model(x_train, y_train, hidden_size=10, learning_rate=0.05, epochs=500)

# Plot accuracy over time
plt.plot(acc_list)
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Over Time")
plt.show()

# Prediction function
def predict(x, w1, b1, w2, b2):
    _, _, _, a2 = forward_propagation(x, w1, b1, w2, b2)
    return np.argmax(a2, axis=1)

# Evaluate model on test data
y_pred_test = predict(x_test, w1, b1, w2, b2)
y_true_test = np.argmax(y_test, axis=1)
test_accuracy = np.mean(y_pred_test == y_true_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Evaluate single sample by user input
def evaluate_user_input(w1, b1, w2, b2, scaler):
    print("\nEnter Sepal and Petal measurements:")
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width = float(input("Petal Width (cm): "))

    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    user_input_scaled = scaler.transform(user_input)

    prediction = predict(user_input_scaled, w1, b1, w2, b2)

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    print(f"Predicted Class: {classes[prediction[0]]}")
evaluate_user_input(w1,b1,w2,b2,scaler)