from PIL import Image
import numpy as np
import genData as gD
import MCM_channel_model as Channel
import tensorflow as tf
 
#------------Parameter for system-------------------
N = 1024   # Number of subcarrier
m = 8     # Bit per symbol
M = 2**m     # Size of one hot symbol
R = m/N
SNR_train_dB = 25
B = 10e6

# Parameter for MonteCarlo
rho = np.array([1, 0.1345, 0.1357])   #discrete multi-path channel profile
N_P = len(rho)
number_of_summations = 50
fD_max = 1024
u = np.random.rand(N_P, number_of_summations)
snr_train = 10**(SNR_train_dB/10.0)
noise_std = np.sqrt(1/(2*R*snr_train))

image_path = 'D:/NamKhanh/LTE-R_Python/parrot.jpg'
image = Image.open(image_path).convert("L") #convert to grayscale
srcImage = np.array(image)

srcSize = srcImage.shape
print(srcSize[0])

srcData = srcImage.reshape(-1,1)
srcOnehot = gD.convert_to_onehot(256, srcData)
print(srcOnehot.shape)
res = gD.reverse_data_from_onehot(srcOnehot)
y = res.reshape(820,-1)
print(y.shape)