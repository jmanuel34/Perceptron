import numpy as np

class Perceptron:
    def __init__(self, ninput, noutput):
        self.ninput = ninput
        self.noutput = noutput
        self.w = np.random.rand(ninput, noutput) - 0.5
        self.b = np.random.rand(noutput) - 0.5

    def forward(self, x):  # propaga un vector x y devuelve la salida
        # a implementar
        #neta = np.product(x[0][0], self.w) + np.product(x[0][1], self.w) + self.b
        neta = np.matmul(x, self.w)+self.b
        return np.piecewise(neta, [neta < 0, neta >= 0], [0, 1])

    def update(self, x, d, alpha):  # realiza una iteración de entrenamiento Data, labels, alpha
        s = self.forward(x)  # propaga
        self.w = self.w+ (alpha * x.reshape(-1, 1) * (d-s))
        # calcula actualización
        # a implementar

    def RMS(self, X, D):
        S = self.forward(X)
        return np.sqrt(np.mean(np.square(S - D)))

    def accuracy(self, X, D):
        S = self.forward(X)
        errors = np.mean(np.abs(D - S))
        return 1.0 - errors

    def info(self, X, D):
        print('     RMS: %6.5f' % self.RMS(X, D))
        print('Accuracy: %6.5f' % self.accuracy(X, D))

    def train(self, X, D, alpha, epochs, trace=0):
        for e in range(1, epochs + 1):
            for i in range(len(X)):
                self.update(X[i], D[i], alpha)
            if trace != 0 and e % trace == 0:
                print('\n   Epoch: %d' % e)
                self.info(X, D)

# entrena para la OR
# 2 entradas y una salida
p = Perceptron(2,1)

# or
data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
labels = np.array([[0.0], [1.0], [1.0], [1.0]])

p.info(data, labels)
p.train(data, labels, 1, 50, 10)