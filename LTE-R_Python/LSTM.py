import numpy as np
import tensorflow as tf
import MCM_channel_model as Channel
import genData as gD
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image
mat = scipy.io.loadmat('snrCompare.mat')


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

# Parameter for AE_LTE-R
pre_trained = False
Nofdm = 128
snr_train = 10**(SNR_train_dB/10.0)
noise_std = np.sqrt(1/(2*R*snr_train))

train_data,_ = gD.gen_data_one_hot(M, 4200)
label_data = train_data
test_size = 100000

'''Build AE-Model'''
#Encoder
Inputs = tf.keras.Input(shape=(M,))
X = tf.keras.layers.Dense(2*N, use_bias= True, activation= 'relu')(Inputs)
X = tf.keras.layers.LayerNormalization()(X)
Z = tf.keras.layers.Reshape((-1,2))(X)

#Channel
Y = tf.keras.layers.Lambda(lambda x: Channel.LTE_R_Channel(x, u, 0, number_of_summations, 1/B, fD_max, rho, noise_std))(Z)

#Decoder
X_rev = tf.keras.layers.Flatten()(Y)
X_tune = tf.keras.layers.Dense(2*N, use_bias= True, activation= 'sigmoid')(X_rev)
X_tune = tf.keras.layers.Dense(M, use_bias= True, activation= 'softmax')(X_tune)

AE_model = tf.keras.Model(inputs=Inputs,outputs=X_tune)


Enc_model = tf.keras.Model(inputs=Inputs,outputs=Z)


Dec_model = tf.keras.Model(inputs= Y,outputs = X_tune)


#model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
#model.fit(x= train_data,y= label_data,batch_size = 5, epochs = 200)
if pre_trained:
    AE_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'mean_squared_error')
    AE_model.fit(train_data, label_data, epochs=20, batch_size=128, verbose=2)
    AE_model.save_weights(r'D:\NamKhanh\LTE-R_Python\models\AE_model_snr20.h5')
    Enc_model.save_weights(r'D:\NamKhanh\LTE-R_Python\models\Enc_model_snr20.h5')
    Dec_model.save_weights(r'D:\NamKhanh\LTE-R_Python\models\Dec_model_snr20.h5')
else:
    Enc_model.load_weights(r'D:\NamKhanh\LTE-R_Python\models\Enc_model_snr20.h5')
    Dec_model.load_weights(r'D:\NamKhanh\LTE-R_Python\models\Dec_model_snr20.h5')
    AE_model.load_weights(r'D:\NamKhanh\LTE-R_Python\models\AE_model_snr20.h5')

#Prepocess Image-------------------
    # image_path = 'D:/NamKhanh/LTE-R_Python/Car.jpg'
    image_path = 'D:/NamKhanh/LTE-R_Python/high_speed_train_original.jpg'
    imgSrc = Image.open(image_path)
    #imgSrc.show()

    image = Image.open(image_path).convert("L") #convert to grayscale
    srcImage = np.array(image)
    #print(srcImage)

    srcSize = srcImage.shape
    source = srcImage.reshape(-1,1)
    test_data = gD.convert_to_onehot(256, srcData)

test_bit = (((source[:,None] & (1 << np.arange(m)))) > 0).astype(int)

#print(test_data)
'''
#sym_errors = (source != pred_output).astype(int).sum()


EbNodB_range = list(np.linspace(-10, 25,8 ))
BER = [None] * len(EbNodB_range)
SER = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10 ** (EbNodB_range[n] / 10.0)
    noise_std_test = np.sqrt(1/(2*R*EbNo))

    Z = Enc_model.predict(test_data)
    Y = Channel.LTE_R_Channel(Z, u, 0, number_of_summations, 1/B, fD_max, rho, noise_std_test)
    X_tune = Dec_model.predict(Y)
    pred_output = np.argmax(X_tune, axis=1)
    re_bit = (((pred_output[:, None] & (1 << np.arange(m)))) > 0).astype(int)
    bit_errors = ((re_bit != test_bit).sum())
    sym_errors = (source != pred_output).astype(int).sum()
    BER[n] = bit_errors / test_size / m
    SER[n] = sym_errors/test_size
    print('SNR:', EbNodB_range[n], 'BER:', BER[n],'SER',SER[n])
    print(source)
    print(pred_output)

ser_linear = mat["ser_linear"][0]
ser_lstm = mat["ser_lstm"][0]
ser_dnn = mat["ser_DNN"][0]

ber = BER
ser = SER
plt.plot(EbNodB_range, ser,'yo--', label='AE_E2E')
plt.plot(EbNodB_range, ser_lstm,'ro--', label='LINEAR')
plt.plot(EbNodB_range, ser_dnn,'bo--', label='DNN-aid')
plt.plot(EbNodB_range, ser_linear,'go--', label='LSTM-aid')
plt.xlabel("SNR(dB)")
plt.ylabel("SER")
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()


#Loop of test
test_value_snr = [0, 2, 4, 5, 10]
for test_value in test_value_snr:
    #Multimedia-----------------------------------------------------

    snr_test_dB = test_value
    snr_test = 10**(snr_test_dB/10.0)
    noise_std_test = np.sqrt(1/(2*R*snr_test))

    #Prepocess Image-------------------
    # image_path = 'D:/NamKhanh/LTE-R_Python/Car.jpg'
    image_path = 'D:/NamKhanh/LTE-R_Python/high_speed_train_original.jpg'
    imgSrc = Image.open(image_path)
    imgSrc.show()

    image = Image.open(image_path).convert("L") #convert to grayscale
    srcImage = np.array(image)
    print(srcImage)

    srcSize = srcImage.shape
    srcData = srcImage.reshape(-1,1)
    srcOnehot = gD.convert_to_onehot(256, srcData)


    #Transmit SrcData-------------------

    Z = Enc_model.predict(srcOnehot)
    Y = Channel.LTE_R_Channel(Z, u, 0, number_of_summations, 1/B, fD_max, rho, noise_std_test)
    X_tune = Dec_model.predict(Y)


    #Process Received Data--------------
    print("Hello")
    X_np = np.array(X_tune)

    # Reverse to matrix from onehot ouput data
    resData = gD.reverse_data_from_onehot(X_np)
    resImage = resData.reshape(srcSize[0],-1)


    # Save ouput image in .png file
    fig, ax = plt.subplots()
    ax.imshow(resImage, cmap='gray')  
    img_name = 'output_high_speed_train_{}dBm.png'.format(test_value)
    plt.savefig(img_name)


    # Open the PNG file
    #image_path = 'D:/NamKhanh/LTE-R_Python/output_high_speed_train_3dBm.png'
    image_path = 'D:/NamKhanh/LTE-R_Python/' + img_name
    img = Image.open(image_path)

    # Display the image
    img.show()



'''