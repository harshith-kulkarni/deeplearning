# Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

# Feature-target split
x = df.drop("medv", axis=1)
y = df["medv"].values

# Scale features
sc_x = StandardScaler()
sc_y = StandardScaler()
x_scaled = sc_x.fit_transform(x)
y_scaled=sc_y.fit_transform(y.reshape(-1,1))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# ReLU activation and derivatives
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Loss functions
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# Initialize weights
def parameter(input_size, hidden_size, output_size):
    np.random.seed(42)
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2

# Forward propagation
def forward_propogation(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = z2
    return z1, a1, z2, a2

# Backward propagation
def backward_propogation(x, y, z1, a1, a2, w2):
    m = x.shape[0]
    dz2 = mse_derivative(y.reshape(-1, 1), a2)
    dw2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
    dw1 = (1 / m) * np.dot(x.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
    return dw1, db1, dw2, db2

# Update weights
def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    return w1, b1, w2, b2

# Train model
def train_model(x, y, hidden_size, learning_rate, epochs):
    lis = []
    input_size = x.shape[1]
    output_size = 1
    w1, b1, w2, b2 = parameter(input_size, hidden_size, output_size)
    for i in range(epochs):
        z1, a1, z2, a2 = forward_propogation(x, w1, b1, w2, b2)
        cost = mse_loss(y.reshape(-1, 1), a2)
        dw1, db1, dw2, db2 = backward_propogation(x, y, z1, a1, a2, w2)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)
        if i % 10 == 0:
            print(f"epoch {i}, cost={cost}")
            lis.append(cost)
    return w1, b1, w2, b2, lis

# Predict using trained model
def predict(x, w1, b1, w2, b2):
    _, _, _, a2 = forward_propogation(x, w1, b1, w2, b2)
    return a2
w1, b1, w2, b2, lis = train_model(x_scaled, y_scaled, hidden_size=10, learning_rate=0.01, epochs=1000)



plt.plot(lis)
plt.xlabel("Epochs (x100)")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss over Epochs")
plt.show()
# Train the model


def evaluate_model(scaler_X, scaler_Y, w1, b1, w2, b2):
    print("\n--- Predict Boston House Price ---")
    column_order = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age','dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
    input_dict = {}

    for col in column_order:
      val = float(input(f"Enter value for {col}: "))
      input_dict[col] = val
    input_array = np.array([[input_dict[col] for col in column_order]])
    input_scaled = scaler_X.transform(input_array)
    predicted_scaled = predict(input_scaled, w1, b1, w2, b2)
    predicted_actual = scaler_Y.inverse_transform(predicted_scaled)
    print(f"\n🏡 Predicted House Price: ${predicted_actual[0][0] * 1000:,.2f}")
evaluate_model(sc_x, sc_y, w1, b1, w2, b2)
