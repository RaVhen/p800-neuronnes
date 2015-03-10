import NNTP

X=
Y=

# create a network with two input, two hidden, and one output nodes

n = NNTP.NN(11, 50, 1)
#train it with some patterns
print "Starting bath training"
n.train(X,Y,iterations=500,N=0.015,M=0,Lambda=0)  # Train is with Back Propagation Algorithm
   