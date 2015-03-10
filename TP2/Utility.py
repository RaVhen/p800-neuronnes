import NNTP
import numpy as np
import matplotlib.pyplot as plt

vals = np.loadtxt("winequality-white.csv",delimiter=";")

X= vals[::, :11]
Y= vals[::, 11:].ravel()
print(X)
print(Y)

indice = list()
val = list()


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

# create a network with two input, two hidden, and one output nodes

for i in my_range(5, 20, 5):
	n = NNTP.NN(11, i, 1) #couche cachee
	#train it with some patterns
	#print "Starting bath training"
	val.appened(n.train(X,Y,iterations=500,N=0.015,M=0,Lambda=0))  # Train is with Back Propagation Algorithm
	print("Pour "+str(i)+" erreur finale de "+str(e))
	indice.appened(i)

plt.plot(indice, val)
plt.show()

# TO DO
# Tracer des courbes en fct du nombre de neuronnes sur la couche cachee
# 
