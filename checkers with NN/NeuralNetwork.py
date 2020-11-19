import math

import numpy as np

from checkers.constants import *


class NN:
    # CREATING NEW UNIVERSAL NN CREATOR
    def __init__(self, input_size, hidden, output):
        # nodes
        self.layers = []
        self.activations = []
        self.errors = []
        self.layers.append(np.zeros(input_size))
        np.append(self.layers[0], np.ones(1))
        self.errors.append(np.zeros(input_size))
        self.activations.append('NONE')
        for i in hidden:
            self.layers.append(np.zeros(i[0]))
            np.append(self.layers[len(self.layers) - 1], [1])
            self.errors.append(np.zeros(i[0]))
            self.activations.append(i[1])
        self.layers.append(np.zeros(output[0]))
        self.errors.append(np.zeros(output[0]))
        self.activations.append(output[1])
        # weights
        self.weights = []
        self.delta = []
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.uniform(-NN_VARIABILITY, NN_VARIABILITY,
                                                  (self.layers[i + 1].shape[0], self.layers[i].shape[0] + 1)))
            self.delta.append(np.zeros((self.weights[i].shape)))
        # print(self.weights)
        self.reward = 0
        self.choices = []
        self.data = []

    def sigmoid(self, x):
        if x > 20:
            return 0.999
        if x < -20:
            return 0.001
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        if x <= 0:
            return RELUMINUS * x
        else:
            return RELUPLUS * x

    def sigderivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def reluderivative(self, x):
        if x <= 0:
            return RELUMINUS
        else:
            return RELUPLUS

    def feedforward(self, data):
        # print(data)
        self.layers[0] = np.append(data, 1)
        for i in range(1, len(self.layers)):
            self.layers[i] = (self.weights[i - 1]).dot(self.layers[i - 1])

            if self.activations[i] == 'RELU':
                self.layers[i] = np.array([self.relu(x) for x in self.layers[i]])
            elif self.activations[i] == 'SIGMOID':
                self.layers[i] = np.array([self.sigmoid(x) for x in self.layers[i]])

            if i != len(self.layers) - 1:
                self.layers[i] = np.append(self.layers[i], 1)

    def softmax(self):
        maks = sum(self.layers[len(self.layers) - 1])
        for i in range(len(self.layers[len(self.layers) - 1])):
            self.layers[len(self.layers) - 1][i] /= maks
            if math.isnan(self.layers[len(self.layers) - 1][i]):
                print(self.layers[len(self.layers) - 1])
        # print(self.layers[len(self.layers)-1])

    def out(self):
        # self.softmax()
        pool = []
        for j in range(len(self.layers[len(self.layers) - 1])):
            for i in range(int(self.layers[len(self.layers) - 1][j] * 100)):
                pool.append(j)
        choice = np.random.choice(pool)
        self.choices.append(choice)
        return choice

    def rate(self, data):
        self.data.append(data)
        self.feedforward(data)
        # print(self.layers[len(self.layers) - 1][0])
        return self.layers[len(self.layers) - 1][0]

    def processing(self, data):
        self.data.append(data)
        self.feedforward(data)
        return self.out()

    def learning(self):
        if len(self.data):
            # clearing data
            for i in range(len(self.errors)):
                self.errors[i] = np.zeros(self.errors[i].shape)

            for i in range(len(self.delta)):
                self.delta[i] = np.zeros((self.weights[i].shape))

            for (exp, label) in zip(self.data, self.choices):
                self.feedforward(exp)
                # creating error vector
                if self.reward > 0:
                    for i in range(len(self.layers[len(self.layers) - 1])):
                        if label == i:
                            self.errors[len(self.errors) - 1][i] = self.layers[len(self.layers) - 1][i] - 1
                        else:
                            self.errors[len(self.errors) - 1][i] = self.layers[len(self.layers) - 1][i]
                else:
                    self.errors[len(self.errors) - 1][label] = self.layers[len(self.layers) - 1][label]

                self.errors[len(self.errors) - 2] = np.transpose(self.weights[len(self.errors) - 2]).dot(
                    self.errors[len(self.errors) - 1][0:len(self.errors[len(self.errors) - 1])])
                for j in range(len(self.errors[len(self.errors) - 2])):
                    if self.activations[len(self.errors) - 2] == 'RELU':
                        self.errors[len(self.errors) - 2][j] *= self.reluderivative(
                            self.layers[len(self.errors) - 2][j])
                    elif self.activations[len(self.errors) - 2] == 'SIGMOID':
                        self.errors[len(self.errors) - 2][j] *= self.sigderivative(self.layers[len(self.errors) - 2][j])

                for i in range(len(self.errors) - 3, 0, -1):
                    self.errors[i] = np.transpose(self.weights[i]).dot(
                        self.errors[i + 1][0:len(self.errors[i + 1]) - 1])
                    for j in range(len(self.errors[i])):
                        if self.activations[i] == 'RELU':
                            self.errors[i][j] *= self.reluderivative(self.layers[i][j])
                        elif self.activations[i] == 'SIGMOID':
                            self.errors[i][j] *= self.sigderivative(self.layers[i][j])

                # print(self.delta[0].shape, self.layers[0].shape, self.errors[1].shape)

                for i in range(len(self.delta)):
                    # print(self.delta[i].shape)
                    for j in range(self.delta[i].shape[0]):
                        for k in range(self.delta[i].shape[1]):
                            # print(i, j, k, '\n', self.errors[i+1].shape, self.layers[i])
                            self.delta[i][j][k] += self.errors[i + 1][j] * self.layers[i][k]

            for i in range(len(self.weights)):
                for j in range(self.weights[i].shape[0]):
                    for k in range(self.weights[i].shape[1]):
                        self.weights[i][j][k] -= (float(self.delta[i][j][k]) / float(
                            len(self.choices))) * LEARNING_RATE * np.abs(self.reward)

            self.data.clear()
            self.choices.clear()

    def show_network(self):
        for weight in self.weights:
            print(weight)
