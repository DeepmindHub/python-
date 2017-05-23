import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors

z=[1]
cur_cost = 0
epochs=2500
np.random.seed(1822)

x = T.dscalar()
fx = T.exp(x**2)
f = theano.function(inputs=[x], outputs=[fx])
fp = T.grad(fx, wrt=x)
fprime = theano.function([x], fp)
x = T.dvector()
y = T.dscalar()
def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.sigmoid(m)
    return h

def layer2(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = m
    return h

def grad_desc(cost, theta):
    alpha = 0.001 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

# theta são os pesos
# 3x3 pois são as duas variáveis + b e layer (2,w)

theta1 = theano.shared(np.array(np.random.rand(3,3), dtype=theano.config.floatX)) # randomly initialize
# theta 2 = entrada 3 adiciona mais um b x 3 (2 var + b) // layer(3,w)

theta2 = theano.shared(np.array(np.random.rand(4,3), dtype=theano.config.floatX))

theta3 = theano.shared(np.array(np.random.rand(4,1), dtype=theano.config.floatX))

hid1 = layer(x, theta1) #hidden layer
hid2 = layer2(hid1, theta2) #hidden layer
out1 = T.sum(layer2(hid2, theta3)) #output layer
fc = (out1 - y)**2 #cost expression

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
        (theta1, grad_desc(fc, theta1)),(theta2, grad_desc(fc, theta2)),
        (theta3, grad_desc(fc, theta3))])
predict = theano.function(inputs=[x], outputs=out1)

import os
os.chdir("/Volumes/16 DOS/Python")
df=pd.read_csv('DadosTeseLogit3.csv',sep=',',header=0)
y=np.array(df[[29]])
y=[item for sublist in y for item in sublist]
x=np.array(df).T


x2=[]
for i in range (0,98):
    x2.append([x[18][i],x[30][i]])
inputs = x2 
exp_y = np.array(y) 

z=[]
for i in range(epochs):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) 
    if i % 1 ==0: 
        z.append(np.mean(cur_cost))
        print('Epoch',i,'| Loss in Training Set = ',z[-1])

plt.plot(1-np.array(z[0:300]),marker='o',linestyle='--',color='g')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Neural Network Accuracy')
plt.show()

w0=[]
for i in range (0,98):
    w0.append(predict([x[18][i],x[30][i]]))

print('Accuracy=',1-np.mean(abs(np.array(y)-w0)))