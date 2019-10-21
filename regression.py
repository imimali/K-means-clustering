'''
Created on Sep 22, 2017

@author: Imi
'''
from math import *


class Regression():
    def __init__(self, learning_rate, no_iter):
        self.alpha = learning_rate
        self.noIter = no_iter
        self.weights = []

    def read_data(self, fname):
        f = open(fname, 'r')
        line = f.readline().strip()
        xs = []
        y = []
        while line != '':
            data_row = line.split(' ')
            y.append(float(data_row[len(data_row) - 1]))
            data_row = data_row[:len(data_row) - 1]
            x_row = [1.0]
            for i in range(len(data_row)):
                x_row.append(float(data_row[i]))
            xs.append(x_row)
            line = f.readline().strip()
        self.weights = [0 for _ in range(len(xs[1]))]
        return xs, y

    def normalize_data(self, no_examples, no_features, train_data):
        mu = []
        sigma = []
        for j in range(no_features):
            summ = 0.0
            for i in range(no_examples):
                summ += train_data[i][j]
            mean = summ / no_examples
            square_sum = 0.0
            for i in range(no_examples):
                square_sum += (train_data[i][j] - mean) ** 2
            deviation = sqrt(square_sum / no_examples)
            mu.append(mean)

            try:
                for i in range(no_examples):
                    train_data[i][j] = (train_data[i][j] - mean) / deviation

            except ZeroDivisionError:
                for i in range(no_examples):
                    train_data[i][j] = 1
            if deviation == 0:
                deviation = 1
            sigma.append(deviation)
        return mu, sigma

    def learning(self, xs, y):
        self.weights = [0 for _ in range(len(xs[1]))]
        for k in range(self.noIter):
            predictions = []
            for xi in xs:
                pred = 0
                for j in range(len(xi)):
                    pred += xi[j] * self.weights[j]
                predictions.append(pred)
            # print(predictions)
            global_error = 0
            for v in range(len(xs[0])):

                for p in range(len(xs)):
                    global_error += (predictions[p] - y[p]) ** 2
            print(global_error)
            # global_error=0
            # print(global_error,' is the error')
            # print(self.weights)
            for j in range(len(self.weights)):

                sum_gr = 0
                for i in range(len(xs)):
                    sum_gr += (predictions[i] - y[i]) * xs[i][j]
                #   print(self.alpha*sum_gr*(1/len(X)))

                self.weights[j] = self.weights[j] - self.alpha * sum_gr * (1 / len(xs))
            # print('dob***************************************dob')
            # print(self.weights)
            # print('dob***************************************dob')

    def predict(self, x):
        pred = self.weights[0]
        for i in range(len(self.weights) - 1):
            pred += self.weights[i + 1] * x[i]
        return pred


reg = Regression(0.00000058, 400)
X, y = reg.read_data('examData.txt')
reg.learning(X, y)
print(reg.predict([30, 10, 10]))
'''
reg2=Regression(0.01,1500)
X,y=reg2.readData('regressionData.txt')
mu,sigma=reg2.normalizeData(len(X),len(X[0]) ,X)
mu[0]=0
sigma[0]=1
reg2.learning(X, y)
print(reg2.weights)
ex=[4203,5]
for i in range(len(ex)):
    ex[i]=(ex[i]-mu[i])/sigma[i]
print(reg2.predict(ex))
'''
