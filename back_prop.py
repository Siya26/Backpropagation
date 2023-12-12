import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def relu(x):
    return np.maximum(0, x)

def normalize_data(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

lr = 0.01

input = np.array([1, 2, 3, 4])
input = np.reshape(input, (4, 1))

weight1 = np.random.rand(2, 4)
bias1 = np.zeros([2, 1])
weight2 = np.random.rand(2, 2)
bias2 = np.zeros([2, 1])

label = np.array([1, 0])
label = np.reshape(label, (2, 1))

input = normalize_data(input)

for i in range(100):
    hidden = np.dot(weight1, input) + bias1
    hidden = sigmoid(hidden)

    output = np.dot(weight2, hidden) + bias2
    output = softmax(output)

    loss = np.log(output) * label
    loss = -np.sum(loss)

    if(i % 10 == 0):
        print(loss)

    dL = output - label

    def back(dL, layer, hidden):
        dA = dL * layer * (1 - layer)
        dw = np.dot(dA, hidden.transpose())

        l = []
        l.append(dA)
        l.append(dw)
        return l

    dA, dw = back(dL, output, hidden)
    weight2 = weight2 - lr * dw
    bias2 = bias2 - lr * dA

    dL = np.dot(weight2.transpose(), dA)
    dA, dw = back(dL, hidden, input)
    weight1 = weight1 - lr * dw
    bias1 = bias1 - lr * dA