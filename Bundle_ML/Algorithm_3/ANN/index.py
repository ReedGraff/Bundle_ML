import numpy as np
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
# General Idea

def ANN(self):
    # Orig
    """
    data = np.array(self.data)
    m, n = data.shape
    print(data.shape)
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:]
    X_dev = X_dev / 255.

    data_train = data[1000:].T
    Y_train = data_train[0]
    X_train = data_train[1:]
    X_train = X_train / 255.
    _,m_train = X_train.shape
    """

    
    # Split Data
    X_train, x_test, Y_train, y_test = clean(self.data)
    X_train = X_train.T / 255

    # https://docs.paperspace.com/machine-learning/wiki/weights-and-biases
    # n = number of inputs
    # Output = Sum(Weight(0->n) * Input(0->n)) + Bias

    # Weight is going to be a 2D array of weights... Where there are X rows, with Y values in each row, where X is the number of nodes in the new layer, and Y is the number of nodes in the previous layer
    
    # https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
    # Apply Activiation function, in this case ReLU, but commonly:
    # - Rectified Linear Activation (ReLU)
    # - Logistic (Sigmoid)
    # - Hyperbolic Tangent (Tanh)
    
    # https://www.quora.com/Why-do-neural-networks-need-an-activation-function
    # We need this because it forces outputs to be a sigmoid, and no longer linear

    ### 3 Parts
    # Part 1 (Forward Propogation)
    # Part 2 (Backward Propogation)
    # Part 3 (Update Parameters)

    # Running:
    # print(len(X_train[0]))

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
    




# Training the model
def init_params():
    # np.random.rand(x, y) returns a 2D array with the shape (x, y), in which every number is a value from 0 to 1
    
    W1 = np.random.rand(10, 784) - 0.5 # Weight
    b1 = np.random.rand(10, 1) - 0.5 # Bias
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    # Stands for Rectified Linear Unit
    # All this does is make all values below 0 = 0... AKA
    # np.maximum([5, 1, -3, 0, 7], 0) = [5, 1, 0, 0, 7]
    return np.maximum(Z, 0)

def softmax(Z):
    # np.exp() simply iterates through an array, and returns an array where each value is e^x (where x is the previous value)... AKA
    # np.exp([5, 1, 0, 3]) = [148.4131591, 2.71828183, 1., 20.08553692]
    # sum() simply returns the sum of the values...
    # Therefore, the exponent divided by the sum of all of the exponents returns the values proportion to the dataset
    # AKA, it converts a vector of numbers into a vector of probabilities, where the probabilities of each value are proportional to the relative scale of each value in the vector
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    # W1 = Weight 1
    # b1 = bias 1
    # W2 = Weight 2
    # b1 = bias 2
    # X = X_Train (AKA inputs)
    
    # This function returns the dot product of two arrays. For 2-D vectors, it is the equivalent to matrix multiplication. 
    # For 1-D arrays, it is the inner product of the vectors.
    # Example:
    # np.dot(np.array([2, 3]), np.array([3, 4])) = (2*3 + 3*4) = 18
    # np.dot(np.array([1, 2]), np.array([3, 4])) = (1*3 + 2*4) = 11

    # a = np.array([[1,2],[3,4]]) 
    # b = np.array([[11,12],[13,14]]) 
    # np.dot(a,b)
    # [[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]
    # [[37, 40], [85, 92]] 


    Z1 = W1.dot(X) + b1
    # Z1 = outputs at each node on the first layer
    A1 = ReLU(Z1)
    # A1 = activated outputs at each node
    Z2 = W2.dot(A1) + b2
    # Z2 = outputs at each node on the second layer
    A2 = softmax(Z2)
    # SoftMaxed outputs at each node, AKA the predicted values...
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    # derivative of the ReLU function
    # Returns an array where true when the value is greater than 0, and false when less than or equal to 0
    return Z > 0

def one_hot(Y):
    # Y = Y_Train (AKA inputs)

    # np.zeros(x, y) returns a 2D array with the shape (x, y), in which every number is 0.
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1

    # Changes columns to rows
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

    # This creates a 2d array in which each row has a 1 at the index of the value... Example:
    # one_hot([5, 4, 2, 1]) returns this:
    # [
    # [0. 0. 0. 0.]
    # [0. 0. 0. 1.]
    # [0. 0. 1. 0.]
    # [0. 0. 0. 0.]
    # [0. 1. 0. 0.]
    # [1. 0. 0. 0.]
    # ]

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = len(X[0])
    
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    # dZ2 = error at node 2
    # error = A2(our predictions) - one_hot_Y(our values)
    
    # Now find how much w and b contributed to that error
    dW2 = 1 / m * dZ2.dot(A1.T)
    # dW2 = error contributed by W2
    db2 = 1 / m * np.sum(dZ2)
    # db2 = error contributed by b2 (AKA average error)


    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    # dZ2 = error at node 2
    # error = W2*dZ2 * inverse activation

    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # alpha = learning rate

    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * db1)

    W2 = W2 - (alpha * dW2)
    b2 = b2 - (alpha * db2)
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2








# Testing the Model / Using the model
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()









# Not Relevant
def clean(data):
    output_vals = np.array(data["label"])
    input_vals = np.array(data.loc[ : , data.columns != 'label'])

    x_train, x_test, y_train, y_test = train_test_split(input_vals, output_vals)

    return train_test_split(input_vals, output_vals)

