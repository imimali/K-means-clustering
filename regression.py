'''
Created on Sep 22, 2017

@author: Imi
'''
from math import *


class Regression():
    def __init__(self, learningRate, noIter):
        self.alpha = learningRate
        self.noIter = noIter
        self.weights = []

    def readData(self, fname):
        f = open(fname, 'r')
        line = f.readline().strip()
        X = []
        y = []
        while line != '':
            dataRow = line.split(' ')
            y.append(float(dataRow[len(dataRow) - 1]))
            dataRow = dataRow[:len(dataRow) - 1]
            xRow = [1.0]
            for i in range(len(dataRow)):
                xRow.append(float(dataRow[i]))
            X.append(xRow)
            line = f.readline().strip()
        self.weights = [0 for k in range(len(X[1]))]
        return X, y

    def normalizeData(self, noExamples, noFeatures, trainData):
        mu = []
        sigma = []
        for j in range(noFeatures):
            sum = 0.0
            for i in range(noExamples):
                sum += trainData[i][j]
            mean = sum / noExamples
            squareSum = 0.0
            for i in range(noExamples):
                squareSum += (trainData[i][j] - mean) ** 2
            deviation = sqrt(squareSum / noExamples)
            mu.append(mean)

            try:
                for i in range(noExamples):
                    trainData[i][j] = (trainData[i][j] - mean) / deviation

            except ZeroDivisionError:
                for i in range(noExamples):
                    trainData[i][j] = 1
            if deviation == 0:
                deviation = 1
            sigma.append(deviation)
        return mu, sigma

    def learning(self, X, y):
        self.weights = [0 for k in range(len(X[1]))]
        for k in range(self.noIter):
            predictions = []
            for xi in X:
                pred = 0
                for j in range(len(xi)):
                    pred += xi[j] * self.weights[j]
                predictions.append(pred)
            # print(predictions)
            globalError = 0
            for v in range(len(X[0])):

                for p in range(len(X)):
                    globalError += (predictions[p] - y[p]) ** 2
            print(globalError)
            # globalError=0
            # print(globalError,' is the error')
            # print(self.weights)
            for j in range(len(self.weights)):

                sumGr = 0
                for i in range(len(X)):
                    sumGr += (predictions[i] - y[i]) * X[i][j]
                #   print(self.alpha*sumGr*(1/len(X)))

                self.weights[j] = self.weights[j] - self.alpha * sumGr * (1 / len(X))
            # print('dob***************************************dob')
            # print(self.weights)
            # print('dob***************************************dob')

    def predict(self, x):
        pred = self.weights[0]
        for i in range(len(self.weights) - 1):
            pred += self.weights[i + 1] * x[i]
        return pred


reg = Regression(0.00000058, 400)
X, y = reg.readData('examData.txt')
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
