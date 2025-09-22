import matplotlib.pyplot as plt
import numpy as np
import genData as gD
x = [12,32,34,54,65,76,12]
y = gD.convert_to_onehot(256,x)
z = gD.reverse_data_from_onehot(y)
print(z)