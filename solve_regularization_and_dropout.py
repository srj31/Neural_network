# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#initializing the parameters for L layer neural network
def relu(Z):
    A = np.maximum(0,Z)
    return A, Z

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid(Z):
    A = 1/(1+(np.exp(-Z)))
    return A,Z


def sigmoid_backward(dA,cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    
    return dA*(1-s)*s

def initialize_parameters_deep(layers_dims):
    parameters = {}
    L = len(layers_dims);
    
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        
    return parameters
        
#calculating the value of Z in NN
def linear_forward_prop(A,W,b):
    Z = np.dot(W,A) + b
    cache = (A,W,b)
    return Z,cache



def linear_activation_fprop(A_prev,W,b,activation):
    
    if(activation=='relu'):
        Z,linear_cache = linear_forward_prop(A_prev,W,b)
        A, activation_cache = relu(Z)
        
    if(activation=="sigmoid"):
        Z,linear_cache = linear_forward_prop(A_prev,W,b)
        A,activation_cache= sigmoid(Z)
        
    cache = (linear_cache,activation_cache)
    return A, cache

#creating the whole netword now using the above fprop functions
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2
    
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_fprop(A_prev, parameters["W"+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)
        
    Al, cache = linear_activation_fprop(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    
    return Al, caches
        
#function to compute the cost of the current parameters
def compute_cost(AL,Y):
    
    m = Y.shape[1]
    
    cost = -1/m*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    cost= np.squeeze(cost)
    return cost
    

#computing cost with L2 regularization
def compute_cost_L2regularization(AL,Y,parameters,lambd=0.7):
    m = Y.shape[1]
    cross_entropy_cost = compute_cost(AL,Y)
    L2_regularization_cost = 0
    for l in range(1,len(parameters)):
        L2_regularization_cost = L2_regularization_cost + np.sum(np.square(parameters["W"+str(l)]))
        
    L2_regularization_cost = L2_regularization_cost*lambd/(2*m)
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
    
    
#Now to do the backward propagation 
    

#to do linear backward knowning dZ, and cache
    
def linear_backward(dZ , cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    
    dW = 1/m*np.dot(dZ,cache[0].T)
    db = 1/m*np.sum(dZ,axis = 1, keepdims = True)
    dA_prev = np.dot(cache[1].T,dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache,activation_cache = cache
    if(activation=="relu"):
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
        
    elif(activation=="sigmoid"):
         dZ = sigmoid_backward(dA,activation_cache)
         dA_prev,dW,db = linear_backward(dZ,linear_cache)
 
    
    return dA_prev,dW,db


def L_model_backward(AL,Y,caches,lambd = 0.7):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -np.divide(Y,AL) + np.divide(1-Y,1-AL)
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    linear_cache,activation_cache = current_cache
    grads["dW"+str(L)] =grads["dW"+str(L)] + lambd/m*linear_cache[1]
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp + lambd/m*(current_cache[0][1])
        grads["db" + str(l + 1)] = db_temp
        
    return grads


def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
    return parameters


def L_layer_model(X,Y,layer_dims,learning_rate = 0.095,num_iterations=3000,print_cost=False):
    costs = []
    
    parameters = initialize_parameters_deep(layer_dims)
    
    for i in range(num_iterations):
        AL,caches = L_model_forward(X,parameters)
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X,Y,parameters):
    Yhat,cache = L_model_forward(X,parameters)
    print(Yhat.shape)
    total_cor = 0;
    predictions = []
    for i in range(Yhat.shape[1]):
          maxi = 0
          ans = 0
          
          for j in range(10):
              if Yhat[j][i] > maxi:
                  maxi = Yhat[j][i]
                  ans = j
          if(Y[0][i]==ans):
              total_cor = total_cor + 1
          predictions.append(ans)
    return  Yhat,predictions
        
    
ds1 = pd.read_csv("test.csv")
ds = pd.read_csv("train.csv")
X_test = ds1.iloc[:,:].values
X_test= X_test.T
X_test = X_test/255
layers_dims = [784, 20, 7, 5, 10] #the L layers and its dimensions
y = ds.iloc[:,0].values
y=y.reshape(42000,1).T
y_new = np.zeros((10,y.shape[1]))
for l in range(y.shape[1]):
    y_new[y[0][l]][l] = 1
X = ds.iloc[:,1:].values
X= X.T
X= X/255
X.shape
parameters = L_layer_model(X,y_new, layers_dims, num_iterations = 4000, print_cost = True)
imgid= []
for i in range(X_test.shape[1]):
    imgid.append(i+1)
yhat,predictions = predict(X_test,y,parameters)
output = pd.DataFrame({"ImageId":imgid,"Label":predictions})
output.to_csv("submission.csv",index = False)
