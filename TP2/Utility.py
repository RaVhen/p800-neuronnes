import NNTP
import numpy as np
import matplotlib.pyplot as plt

vals = np.loadtxt("winequality-white.csv",delimiter=";")

X= vals[::, :11]
Y= vals[::, 11:].ravel()
print(X)
print(Y)

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

#root mean square error
def rmse(predicted,observed):
  return np.sqrt(((predicted - observed) ** 2).mean())

def linear_regression(X,Y,epsilon,nbiteration):
  w = np.random.rand(X.shape[1])
  #print(w.shape)
  ord = list()
  abs = list()
  indice = list()
  val = list()
  for i in range (nbiteration):
    prod = np.dot(X,w)           #multiplication de matrice
    #print(prod.shape)
    error = rmse(Y, prod)
    print ("Erreur = ",error)
    w += epsilon*np.dot(X.T,(Y-prod))
    ord.append(error)
    abs.append(i)
  for i in my_range(1, 150, 10):
        print("********Iteration "+str(i)+"**********")
	n = NNTP.NN(11, i, 1) #couche cachee
	val.append(n.train(X,Y,iterations=500,N=0.00015,M=0,Lambda=0))  # Train is with Back Propagation Algorithm
	indice.append(i)
  plt.plot(abs, ord, label="Linear regression")
  plt.plot(indice, val, label="Neural network")
  plt.xlim(0, 150)
  plt.ylim(0, 20)
  plt.legend()
  plt.title("Comparaison entre la regression lineaire et le reseau de neurones")
  plt.show()
  return w

X = np.loadtxt("winequality-white.csv",delimiter=";")
print(X.shape)

vals = X[::, :11]
print(vals.shape)
notes = X[::, 11:].ravel()
print(notes)
print("\n*********START*************\n")
print(linear_regression(vals,notes,0.00000001,1000000))



# create a network with two input, two hidden, and one output nodes

#for i in my_range(5, 20, 5):
#	n = NNTP.NN(11, i, 1) #couche cachee
#	#train it with some patterns
#	#print "Starting bath training"
#	val.append(n.train(X,Y,iterations=500,N=0.015,M=0,Lambda=0))  # Train is with Back Propagation Algorithm
#	print("Pour "+str(i)+" erreur finale de ")
#	indice.append(i)

#plt.plot(indice, val)
#plt.show()

# TO DO
# Tracer des courbes en fct du nombre de neuronnes sur la couche cachee
# 
