import numpy as np
import neurolab as nl


# Define the input file
input_file = 'data/letter.data'

# Define the number of data points
num_data = 50

# String containing all the distinct characters
orig_labels = 'omandig'

# Store number of distinct characters
num_labels = len(orig_labels)

# Define the training and testing parameters
num_train = int(0.9 * num_data)
num_test = num_data - num_train

# Define the dataset extraction parameters
start = 6
end = -1

# Creating the dataset
data = list()
labels = list()
with open(input_file, 'r') as f:
    for line in f.readlines():
        # Split the current line tabwise
        list_vals = line.split('\t')

        # Check if the label is in our ground truth labels
        # If not, we should skip it.
        if list_vals[1] not in orig_labels:
            continue

        # Extract the current label and append it to the main list
        label = np.zeros((num_labels, 1))
        label[orig_labels.index(list_vals[1])] = 1
        labels.append(label)

        # Extract the character vector and append it to the main list
        cur_char = np.array([float(x) for x in list_vals[start:end]])
        data.append(cur_char)

        # Exit the loop once the required dataset has been created
        if len(data) >= num_data:
            break

f.close()

# Convert the data and labels to numpy arrays
data = np.asfarray(data)
labels = np.array(labels).reshape(num_data, num_labels)

# Extract the number of dimensions
num_dims = len(data[0])

# Create a feed forward neural network
nn = nl.net.newff([[0, 1] for _ in range(len(data[0]))], [128, 16, num_labels])

# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd

# Train the network
err = nn.train(data[:num_train, :], labels[:num_train, :],
               epochs=10000, show=100, goal=0.01)

# Predict the output for test inputs
print('\nTesting on unknown data:')
predicted_test = nn.sim(data[num_train:, :])
for i in range(num_test):
    print('\nOriginal:', orig_labels[np.argmax(labels[i])])
    print('Predicted:', orig_labels[np.argmax(predicted_test[i])])
