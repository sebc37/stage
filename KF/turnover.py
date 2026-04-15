import numpy as np
import matplotlib.pyplot as plt

PATH = '/home/s26calme/Documents/code_stage/GOY-main/data_test_precis.dat'
SAVE = '/home/s26calme/Documents/code_stage/KF/'

data = np.loadtxt(PATH,dtype=np.float64)

n=22
k0 = 0.125
lmb = 2.0

K = np.array([k0*lmb**i for i in range(n)],dtype=np.float64)
Tn = np.zeros((10,n),dtype=np.float64)
Ntime = data.shape[0]


for j in range(10):
    for i in range(n):
        Tn[j,i] = 1/(K[i]*np.sqrt(np.mean(data[int((j/10)*Ntime):int(((j+1)/10)*Ntime),2*i]**2 + data[int((j/10)*Ntime):int(((j+1)/10)*Ntime),2*i+1]**2)))

    plt.figure()
    plt.plot(Tn[j,:],'.k',label = f'{j/10} %')
    plt.xlabel("$k_n$")
    plt.ylabel("$T_n$")
    plt.legend()


m = []
M = []
for j in range(10):
    m.append(min(Tn[j,:]))
    M.append(max(Tn[j,:]))

plt.figure()
plt.plot(m,'.b',label="min $T_n$")
plt.plot(M,'.r',label="max $T_n$")
plt.legend()
plt.show()

print(min(m))
print(max(M))

print(np.mean(m))
print(np.mean(M))
#plt.savefig(SAVE + "Turnover")

