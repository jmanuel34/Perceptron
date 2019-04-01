import numpy as np
import math
import sys
class Multilayer:
    def __init__(self, layers):
        self.layers = layers

        # Inicializamos los pesos y los bias
        self.w = []
        self.b = []
        for i in range(len(layers) - 1):
            # numero de entradas, numero de perceptrones layer[i+1]
            self.w.append(np.random.rand(layers[i], layers[i + 1]) - 0.5)
            self.b.append(np.random.rand(layers[i + 1]) - 0.5)          #numero de perceptrones [i+1] dados en el array

        # Lista de salidas intermedias
        self.s = []

    def sigm(self, neta):
        return 1.0 / (1.0 + np.exp(-neta))

    def forward(self, x):  # propaga un vector x y devuelve la salida
#        self.s1 = self.sigm(np.matmul(x, self.w1)+self.b1)
#        self.s2 = self.sigm(np.matmul(self.s1, self.w2) + self.b2)
#        self.s[1] = self.sigm(np.matmul(x, self.w[0]) + self.b[0])

        self.s[0] = self.sigm(np.matmul(x, self.w[0]) + self.b[0])

        for i in range(len(sys.argv) - 1):
            self.s[i+1] = self.sigm(np.matmul(self.s[i], self.w[i+1]) + self.b[i+1])

            return self.s

    # a implementar

    def update(self, x, d, alpha):  # realiza una iteración de entrenamiento
        # a implementar
        s = self.forward(x)  # propaga

        delta3 = (d - self.s2) * self.s2 * (1 - self.s2)
        delta2 = np.matmul(delta3, self.w2.T) * self.s1 * (1 - self.s1)
        delta1 = np.matmul(delta2, self.w1.T) * self.x * (1 - self.x)


        self.w3 = self.w3 + alpha * self.s2.reshape(-1, 1) * delta3
        self.b3 = self.b3 + alpha * delta3

        self.w2 = self.w2 + alpha * self.s1.reshape(-1, 1) * delta2
        self.b2 = self.b2 + alpha * delta2

        self.w1 = self.w1 + alpha * x.reshape(-1, 1) * delta1
        self.b1 = self.b1 + alpha * delta1


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

p = Multilayer([2,2,2,1])



p.info(data, labels)
p.train(data, labels, 0.7, 30000, 3000)

