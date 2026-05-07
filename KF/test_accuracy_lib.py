import numpy as np
import matplotlib.pyplot as plt

PATH_GOY = '/home/s26calme/Documents/code_stage/GOY-main/data_enKF_base.dat'
PATH_LIB = '/home/s26calme/Documents/code_stage/KF/goy_lib_test.npy'

data_goy = np.loadtxt(PATH_GOY,dtype=np.float64)
data_lib = np.load(PATH_LIB,allow_pickle=True)

print(data_goy.shape)
print(data_lib.shape)

for i in range(0,44,2):
    plt.figure()
    plt.plot(data_goy[:,i],label="goy")
    plt.plot(data_lib[:,i],label="lib")
    plt.legend()
plt.show()