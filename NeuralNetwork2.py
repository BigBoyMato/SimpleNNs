import numpy as np


# Initially set NN weights
W1 = np.array([[0.6, 0.4, 0.2, 0.2], [0.0, 0.1, 0.2, 0.2]])
W2 = np.array([1.0, 1.0])


# epoch as a vector of observations (first 4 elements)
# and expected result (last element)
# generated as -> print([bin(i)[2: ].rjust(4,"0").replace("0","-1") for i in range(16)])
epoch = [(-1, -1, -1, -1, -1),
 (1, -1, -1, 1, -1),
 (1, -1, 1, -1, -1),
 (-1, -1, 1, 1, -1),
 (-1, 1, -1, -1, -1),
 (-1, 1, -1, 1, -1),
 (-1, 1, 1, -1, -1),
 (-1, 1, 1, 1, -1),
 (1, -1, -1, -1, -1),
 (1, -1, -1, 1, -1),
 (1, -1, 1, -1, -1),
 (1, -1, 1, 1, 1),
 (1, 1, -1, -1, 1),
 (1, 1, -1, 1, 1),
 (1, 1, 1, -1, 1),
 (1, 1, 1, 1, 1)]


# hyperbolic tangent function as an act function
def act_f(x):
    return 2/(1 + np.exp(-x)) - 1


# hyperbolic tangent function derivative
def act_df(x):
    return 0.5*(1 + x)*(1 - x)


# Neural Network implementation
class NeuralNetwork():
    def __init__(self, W1, W2, epoch):
        self.W1 = W1
        self.W2 = W2
        self.epoch = epoch

    # NN one-step iterate function
    def iterate_once(self, inp):
        sum = np.dot(self.W1, inp)
        out = np.array([act_f(x) for x in sum])

        sum = np.dot(self.W2, out)
        y = act_f(sum)
        return (y, out)

    # NN training function
    def train(self):
        lmd = 0.01  # NN training step
        N = 10000  # amount of NN training iterations
        count = len(self.epoch)
        for k in range(N):
            x = self.epoch[np.random.randint(0, count)]  # random epoch element pick
            y, out = self.iterate_once(x[0:4])  # iteration through NN
            e = y - x[-1]  # error occurrence
            delta = e * act_df(y)  # local gradient
            self.W2[0] = self.W2[0] - lmd * delta * out[0]  # first NN connection adjustment
            self.W2[1] = self.W2[1] - lmd * delta * out[1]  # second NN connection adjustment

            delta2 = W2 * delta * act_df(out)  # vector of 2 local gradient values

            # adjustment of first layer connections
            self.W1[0, :] = self.W1[0, :] - np.array(x[0:4]) * delta2[0] * lmd
            self.W1[1, :] = self.W1[1, :] - np.array(x[0:4]) * delta2[1] * lmd

    # return current epoch
    def CheckEpochAndMakeDecision(self):
        for x in self.epoch:
            y, out = neural_network.iterate_once(x[0:4])
            print(f"NN output value: {y} => {x[-1]}")

            if x[-1] == 1:
                print("person will eat")
            if x[-1] == -1:
                print("person won't eat")


# commands to process NN
neural_network = NeuralNetwork(W1, W2, epoch)
neural_network.train()
neural_network.CheckEpochAndMakeDecision()




