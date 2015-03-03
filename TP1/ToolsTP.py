import math
import numpy as np

first = np.arange(1,11)
second = np.random.randn(10)
third = np.zeros((1,10))

final = np.vstack((first, second, third))

m7 = np.vstack((np.array([1, 2, 3]), final.T))
#print(m7)


def tanh(x):
  return np.tanh(x)


def tanh_deriv(x):
  return (1-tanh(x)**2)

def sigm(x):
  return (1/(1+np.exp(-x)))

def sigm_derivative(x):
  return (sigm(x)*(1-sigm(x)))

#X a 4 colonnes, ci est la ieme colonne, renvoie 3*c1 + 2*c2 - 5*c3 + 1*c4
def polynom(X):
  return(3*X[:,0] + 2*X[:,1] - 5*X[:,2] + X[:,3])
  

#root mean square error
def rmse(predicted,observed):
  return np.sqrt(((predicted - observed) ** 2).mean())

#renvoie un Vecteur de poids W tq [X.W-Y]^2 est minimum
def linear_regression(X,Y,epsilon,nbiteration):
  w = np.random.rand(X.shape[1])
  #print(w.shape)
  for i in range (nbiteration):
    prod = np.dot(X,w)           #multiplication de matrice
    #print(prod.shape)
    error = rmse(Y, prod)
    print ("Erreur = ",error)
    w += epsilon*np.dot(X.T,(Y-prod))
  return w
    
#generer 
x = np.random.rand(10,4)
y = polynom(x)
#print(linear_regression(x,y,0.01,1000))

X = np.loadtxt("winequality-white.csv",delimiter=";")
print(X.shape)

Mat = np.ones((X.shape[0], X.shape[1]-1))
vals = X[::, :11]
print(vals.shape)
notes = X[::, 11:].ravel()
print(notes)
print("\n*********START*************\n")
print(linear_regression(vals,notes,0.00000001,1000))
