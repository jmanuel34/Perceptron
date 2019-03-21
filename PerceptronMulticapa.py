import numpy as np
import math
class Multilayer:
    def __init__(self, ninput, nhidden, noutput):
        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput

        self.w1 = np.random.rand(ninput, nhidden) - 0.5
        self.b1 = np.random.rand(nhidden) - 0.5
        self.w2 = np.random.rand(nhidden, noutput) - 0.5
        self.b2 = np.random.rand(noutput) - 0.5

        self.lRMS = []  # contiene la lista de RMSs para pintarlos
        self.laccuracy = []  # contiene la lista de accuracy
        self.s1=np.empty((2,1))
        self.s2=np.empty((1,1))

    def sigm(self, neta):
        return 1.0 / (1.0 + np.exp(-neta))

    def forward(self, x):  # propaga un vector x y devuelve la salida
        self.s1 = self.sigm(np.matmul(x, self.w1)+self.b1)
        self.s2 = self.sigm(np.matmul(self.s1, self.w2) + self.b2)
        return self.s2

    # a implementar

    def update(self, x, d, alpha):  # realiza una iteración de entrenamiento
    # a implementar
        s = self.forward(x)  # propaga

        delta2= np.dot(d - self.s2, self.s2)*(1-self.s2)
        self.w2 = (self.w2 + np.dot(alpha*self.s1.reshape(-1,1),delta2).reshape(-1, 1))
        delta1 = np.dot(delta2, self.w2.T)*self.s1*(1-self.s1)
        self.w1= self.w1+ (np.dot(alpha,x.reshape(-1,1))*delta1)
        self.b1 = self.b1 + np.dot(alpha, delta1)
        self.b2=self.b2 + np.dot(alpha, delta2)

    def RMS(self, X, D):
        S = self.forward(X)
        return np.sqrt(np.mean(np.square(S - D)))

    def accuracy(self, X, D):
        S = self.forward(X)
        S = np.round(S)
        errors = np.mean(np.abs(D - S))
        return 1.0 - errors

    def info(self, X, D):
        self.lRMS.append(self.RMS(X, D))
        self.laccuracy.append(self.accuracy(X, D))
        print('     RMS: %6.5f' % self.lRMS[-1])
        print('Accuracy: %6.5f' % self.laccuracy[-1])

    def train(self, X, D, alpha, epochs, trace=0):
        self.lRMS = []  # guarda lista de RMSs para pintarlos
        self.laccuracy = []  # guarda lista de accuracy

        for e in range(1, epochs + 1):
            for i in range(len(X)):
                self.update(X[i], D[i], alpha)
            if trace != 0 and e % trace == 0:
                print('\n   Epoch: %d' % e)
                self.info(X, D)


    def one_hot(d):
        num_classes = len(set(d))
        rows = d.shape[0]
        labels = np.zeros((rows, num_classes), dtype='float32')
        labels[np.arange(rows), d.T] = 1
        return labels

# xor
data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
labels = np.array([[0.0], [1.0], [1.0], [0.0]])

p = Multilayer(2,2,1)

p.info(data, labels)
p.train(data, labels, 0.8, 5000, 1000)

