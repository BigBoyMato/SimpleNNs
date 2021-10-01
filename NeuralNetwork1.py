import numpy as np


# initially set NN weights
w11 = [0.6, 0.4, 0.2, 0.2]
w12 = [0.0, 0.1, 0.2, 0.2]
w2 = [1.0, 1.0]


# last layer activate function
def act(x):
    return 1.0 if x >= 1.0 else 0.0


# human-type decision making function
def make_decision(y):
    if y == 1.0:
        print("Person will eat")
    else:
        print("Person won't eat")


# Neural Network implementation
class NeuralNetwork():
    def __init__(self, input_values, w11, w12, w2):
        self.input_values = input_values
        self.weight1 = np.array([w11, w12])
        self.weight2 = w2

        self.sum_hidden = np.dot(0.0, 0.0)
        self.out_hidden = np.array([])
        self.sum_end = np.dot(0.0, 0.0)
        self.y = 0.0

    def train_neural_network(self):
        self.sum_hidden = np.dot(self.weight1, self.input_values)
        print("Sums values on hidden neuron layer: " + str(self.sum_hidden))

        self.out_hidden = np.array([act(input_values) for input_values in self.sum_hidden])
        print("Values of output neurons on hidden neuron layer: " + str(self.out_hidden))

        self.sum_end = np.dot(self.weight2, self.out_hidden)
        self.y = act(self.sum_end)
        print("Output value of Neural Network: " + str(self.y))

        return self.y


# commands to input
is_hungry = int(input("Enter 1 if person is hungry: "))
has_food = int(input("Enter 1 if person has food at home: "))
store_is_near = int(input("Enter 1 if store is near to person: "))
has_money = int(input("Enter 1 if person has money to buy food: "))


# commands to process neural network
input_values = [is_hungry, has_food, store_is_near, has_money]
neural_network = NeuralNetwork(input_values, w11, w12, w2)
make_decision(neural_network.train_neural_network())
