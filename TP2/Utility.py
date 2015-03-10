import NNTP
import numpy as np

vals = np.loadtxt("winequality-white.csv",delimiter=";")

X= vals[::, :11]
Y= vals[::, 11:].ravel()
print(X)
print(Y)


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

# create a network with two input, two hidden, and one output nodes

for i in my_range(5, 45, 5):
	n = NNTP.NN(11, i, 1) #couche cachee
	#train it with some patterns
	#print "Starting bath training"
	e=n.train(X,Y,iterations=500,N=0.015,M=0,Lambda=0)  # Train is with Back Propagation Algorithm
	print(str(i)+" "+str(e))   		

# TO DO
# Tracer des courbes en fct du nombre de neuronnes sur la couche cachee
# 