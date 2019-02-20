import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl


# Generate some training data
min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

# Create data and labels
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

# Define a multilayer neural network with 2 hidden layers
# First hidden layer --> 10 neurons
# Second hidden layer --> 6 neurons
nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])

# Train the neural network
err = nn.train(data, labels, epochs=2000, show=100, goal=0.01)

# Run the neural network on training data points
output = nn.sim(data)
y_pred = output.reshape(num_points)

# Plot the training progress
plt.figure()
plt.plot(err)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')

# Plot the output
x_dense = np.linspace(min_val, max_val, num_points*2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)

plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs. Predicted')
plt.show()
