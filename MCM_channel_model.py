import numpy as np
from math import pi,sin,sqrt,exp
from cmath import exp
import tensorflow as tf

def ComplexLayer(X):
     return np.array(X[:,0])+1j*np.array(X[:,1])

def DeComplexLayer(X):
     return np.append(np.real(X),np.imag(X))

def MCM_channel_model(u, initial_time, number_of_summations, sample_duration, f_dmax, channel_coefficients):
    t = initial_time
    channel_length = len(channel_coefficients)
    h_vector = []
    for k in range(0,channel_length):
        u_k = u[k,:]
        phi = np.array(2*pi*u_k)
        f_d = f_dmax*np.sin(phi)
        h_tem = channel_coefficients[k]* 1/(sqrt(number_of_summations)) * np.sum(np.exp(1j*phi)*np.exp(1j*2*pi*f_d*t))
        h_vector.append(h_tem)
    t_next = initial_time + sample_duration
    h = np.array(h_vector)
    return h, t_next

def LTE_R_Channel(Data_tx, u, init_time, number_of_summations, sample_duration, f_dmax, channel_coefficients, noise_std):
        h_org = np.complex64(np.zeros((1,1024,3)))
        y_real = tf.zeros((1,1024))
        y_imag = tf.zeros((1,1024))    
        for k in range(0,1024):
            h, t_next = MCM_channel_model(u, init_time, number_of_summations, sample_duration, f_dmax, channel_coefficients)
            h_org[0,k,:] = h
            init_time = t_next
        h_real = tf.convert_to_tensor(np.real(h_org))
        h_imag = tf.convert_to_tensor(np.imag(h_org))
        x_real = Data_tx[:,:,0]
        x_imag = Data_tx[:,:,1]
        #gen X_shift
        for ii in range(0,3):
            #x_real = tf.convert_to_tensor(np.append((np.zeros((ii,))), Data_shift[0:1024-ii,0]))
            #x_imag = tf.convert_to_tensor(np.append((np.zeros((ii,))), Data_shift[0:1024-ii,1]))
            y_real =y_real + h_real[:,:,ii]*x_real - h_imag[:,:,ii]*x_imag
            y_imag = y_imag + h_imag[:,:,ii]*x_real + h_real[:,:,ii]*x_imag
        #y_real = tf.reshape(y_real,(1024,-1))
        #y_imag = tf.reshape(y_imag,(1024,-1))
        noise_r = tf.random.normal((1,1024),0,noise_std)
        noise_i = tf.random.normal((1,1024),0,noise_std)
        y_real = y_real + noise_r
        y_imag = y_imag + noise_i
        y = tf.stack([y_real, y_imag], axis =2 )
        print(y.shape)
        return y   #output shape = (1024,2)



'''Test data
x = np.random.rand(1024,2)
print(x)
print(LTE_R_Channel(x,np.random.rand(3,50),0,50,1,1024,[0.53,0.69,0.56], 0.2))'''

