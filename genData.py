import numpy as np

def gen_data_one_hot(M, datasize):
    sourcedata = np.random.randint(M, size= datasize)
    data = []
    for i in sourcedata:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)
    data = np.array(data)
    return data, sourcedata


def convert_to_onehot(M, srcData):
    data = []
    for i in srcData:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)
    data = np.array(data)
    return data

def reverse_data_from_onehot(srsData):
    data = []
    for i in srsData:
        data.append(np.argmax(i))
    data = np.array(data)
    return data

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

'''Test func
x,src = gen_data_one_hot(6, 4)
print(src)
print(x)
print(reverse_data_from_onehot(x))
'''
