from __future__ import division
import numpy as np
import matplotlib.pyplot as plot
import theano
import theano.tensor as T
import sklearn
from sklearn.datasets import fetch_mldata
import sklearn.datasets as sk_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from numpy.random import randn


def _NN__SIGMOID(z):
    return 1/(1+np.exp(-1*z))

def _NN__SIGMOID_PRIME(z):
    return _NN__SIGMOID(z)*(1-_NN__SIGMOID(z))

def _NN__TAN_H(z):
    num = np.exp(z)-np.exp(-1*z)
    denom = np.exp(z)+np.exp(-1*z)
    return num/denom

def _NN__C_DERIV(a, y): return a-y

def shuffle(X,Y):
    if (len(X)!=len(Y)):
        raise Exception("incorrect size %s!=%s"%(len(X),len(Y)))
    shuf_idx = np.arange(len(X))
    X_, Y_ = np.zeros(X.shape), np.zeros(Y.shape)
    np.random.shuffle(shuf_idx)
    for i in range(len(shuf_idx)):
        X_[shuf_idx[i]] = X[i]
        Y_[shuf_idx[i]] = Y[i]
    return X_,Y_


class NN():
    def __init__(self, input_size, learning_rate=0.01, isClassifying=True):
        self.weights = []
        self.biases = []
        self.layer_sizes = [input_size]
        self.act_functions = []
        self.learning_rate = learning_rate
        self.isClassifying = isClassifying

    def n_layers(self):
        return len(self.layer_sizes)

    def addDense(self, size, f=__SIGMOID):
        self.weights.append(randn(self.layer_sizes[-1], size))
        self.biases.append(randn(1, size))
        self.act_functions.append(f)
        self.layer_sizes.append(size)

    def compute(self, x):
        if (len(x)!=self.layer_sizes[0]):
            raise "incorrect layer size " + str(len(x)) + " supposedly : " +str(self.layer_sizes[0])
        Out = x
        for i in range(len(self.weights)):
            (w, b, f) = (self.weights[i], self.biases[i], self.act_functions[i])
            Out = f(np.dot(Out, w)+b)
        return Out.reshape((self.layer_sizes[-1],))

    def fit(self, X, Y, epoch=100, n_minibatch=10, loss_f=mean_squared_error):
        n = len(X)
        for i in range(epoch):
            X,Y = shuffle(X,Y)
            print 'epoch %s/%s'%(i+1, epoch)
            for j in range(0, n, n_minibatch):
                batch_X = X[j: j+n_minibatch]  if (j+n_minibatch < n) else X[j:]
                batch_Y = Y[j: j+n_minibatch]  if (j+n_minibatch < n) else Y[j:]
                self.SGD(batch_X, batch_Y, loss_f)

            acc_score = self.evaluate_loss(batch_X, batch_Y)
            print "\tloss=", 1-acc_score

    def predict(self, X):
        if (self.isClassifying):
            return np.argmax(self.compute(X))
        raise Exception("Cannot predict")

    def evaluate_loss(self, X_train, Y_train):
        X_hat, Y_hat = np.array([np.argmax(self.compute(x)) for x in X_train]), [np.argmax(y) for y in Y_train]
        score = accuracy_score(X_hat, Y_hat)
        return score

    def SGD(self, X, Y, loss_f):
        # (1) output gradient = (dC/da)sigmoid'(NN-output) = delta^L
        # (2) hidden layer gradient = (W^l_jk)(delta^l) (*) sigmoid'(z^(l-1)) = delta^l-1
        # bias -> replace w with b vector
        n_minibatch = len(X)
        accum_dCdw = [np.zeros(_w.shape) for _w in self.weights]
        accum_dCdb = [np.zeros(_b.shape) for _b in self.biases]

        for x,y in zip(X,Y):
            z_stack = []
            a_stack = [x.reshape(1,x.shape[0])]
            a_L = a_stack[0]
            # feed forward
            for i in range(len(self.weights)):
                (w, b) = (self.weights[i], self.biases[i])
                z_stack.append(np.dot(a_L, w) + b)
                a_L = _NN__SIGMOID(z_stack[-1])
                a_stack.append(a_L)
            #backpropagation
            delta_l = _NN__C_DERIV(a_stack.pop(), y)*_NN__SIGMOID_PRIME(z_stack.pop()) #partial_dif = (dC/da)*sigmoid'(z)
            accum_dCdw[-1] += np.dot(a_stack.pop().T, delta_l) #dC/dw = (dC/da)(da/dz)(dz/dw)=(delta_l)(a_i-1)
            accum_dCdb[-1] += delta_l

            for i in range(len(z_stack)):
                (a_i, z_i) = (a_stack.pop(), z_stack.pop())
                # case hidden layer
                delta_l = np.dot(delta_l, self.weights[-i-1].T)*_NN__SIGMOID_PRIME(z_i)
                dCdw = np.dot(a_i.T, delta_l)
                accum_dCdw[-i-2] += dCdw #dCdw
                accum_dCdb[-i-2] += delta_l # dCdb, bias weights = delta

        for w, b, dCdw_i, dCdb_i in zip(self.weights, self.biases, accum_dCdw, accum_dCdb):
            w-=(self.learning_rate/n_minibatch)*dCdw_i
            b-=(self.learning_rate/n_minibatch)*dCdb_i

    # def eval_loss(self, X, Y, loss_f=mean_squared_error):
    #     n = len(X)
    #     for i in range(n):

    def summary(self):
        for i in range(len(self.layer_sizes)):
            print "size=%s"%(self.layer_sizes[i])
            if i < len(self.layer_sizes)-1:
                print "weight=%s bias=%s f=%s"%(self.weights[i].shape, self.biases[i].shape, self.act_functions[i])

if __name__ == '__main__':
    # model = NN(3)
    # model.addDense(5)
    # model.addDense(5)
    # model.addDense(1)
    # model.summary()
    # out = model.compute(np.array([100,100,100]))
    # print out
    print ("===========")
    model = NN(3)
    model.addDense(5)
    model.addDense(5)
    model.addDense(2)
    model.summary()
