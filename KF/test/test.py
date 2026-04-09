from goy import GoyModel
import numpy as np
import matplotlib.pyplot as plt
model = GoyModel()         # paramètres identiques au code original
sh = model.shell_wavenumbers()

# conditions initiales
Xpp = sh**(-1/3)
Ypp = np.full(model.N, 1e-4)
Xp, Yp = Xpp.copy(), Ypp.copy()
n_step = 100100
state = np.zeros((n_step,44))

Xpp, Ypp, Xp, Yp = model.init_fields()

data  = np.loadtxt("/home/s26calme/Documents/code_stage/GOY-main/data_test_precis.dat",dtype=np.float32)
X_ref = data[:, 0::2]   # (n_snap, N)
Y_ref = data[:, 1::2]

# state[0,0::2] = X_ref[0,:]
# state[0,1::2] = Y_ref[0,:]
state[0,0::2],state[1,0::2] = Xpp,Xp
state[0,1::2],state[1,1::2] = Ypp,Yp


def NL(x_past_real,x_past_imag):
    
    eps = 0.5
    lmb = 2.0
    n = np.shape(x_past_real)[0]
    NL_re = np.zeros(n)
    NL_im = np.zeros(n)    
    
    NL_re[0] = x_past_real[2]*x_past_imag[1] + x_past_imag[2]*x_past_real[1]
    NL_re[1] = x_past_real[3]*x_past_imag[2] + x_past_imag[3]*x_past_real[2] - (eps/lmb)*K[1]*(x_past_real[0]*x_past_imag[2] + x_past_imag[0]*x_past_real[2])

    NL_im[0] = x_past_real[2]*x_past_real[1] - x_past_imag[2]*x_past_imag[1]
    NL_im[1] = x_past_real[3]*x_past_real[2] - x_past_imag[3]*x_past_imag[2] - (eps/lmb)*K[1]*(x_past_real[0]*x_past_real[2] - x_past_imag[0]*x_past_imag[2])

    for i in range(2,n-2):
        NL_im[i]  =  K[i]*(x_past_real[i+1]*x_past_imag[i+2] + x_past_imag[i+1]*x_past_real[i+2]) 

        -K[i]*(eps/lmb)*(x_past_real[i-1]*x_past_imag[i+1] + x_past_imag[i-1]*x_past_real[i+1]) 
                
        + K[i]*((eps-1)/lmb**2)*(x_past_real[i-2]*x_past_imag[i-1] + x_past_imag[i-2]*x_past_real[i-1])

        

        NL_re[i] = K[i]*(x_past_real[i+1]*x_past_real[i+2] - x_past_imag[i+1]*x_past_imag[i+2]) 
        
        -K[i]*(eps/lmb)*(x_past_real[i-1]*x_past_real[i+1] - x_past_imag[i-1]*x_past_imag[i+1]) 
            
        + K[i]*((eps-1)/lmb**2)*(x_past_real[i-2]*x_past_real[i-1] - x_past_imag[i-2]*x_past_imag[i-1]) 

    NL_re[n-2] = -(eps/lmb)*K[n-2]*(x_past_real[n-3]*x_past_imag[n-1] + x_past_imag[n-3]*x_past_real[n-1])  
    + K[n-2]*((eps-1)/lmb**2)*(x_past_real[n-4]*x_past_imag[n-3] + x_past_imag[n-4]*x_past_real[n-3])
    
    NL_re[n-1] = K[n-1]*((eps-1)/lmb**2)*(x_past_real[n-3]*x_past_imag[n-2] + x_past_imag[n-3]*x_past_real[n-2])
    
    NL_im[n-2] = -(eps/lmb)*K[n-2]*(x_past_real[n-3]*x_past_real[n-1] + x_past_imag[n-3]*x_past_imag[n-1])
    + K[n-2]*((eps-1)/lmb**2)*(x_past_real[n-4]*x_past_real[n-3] + x_past_imag[n-4]*x_past_imag[n-3])

    NL_im[n-1] = K[n-1]*((eps-1)/lmb**2)*(x_past_real[n-3]*x_past_real[n-2] - x_past_imag[n-3]*x_past_imag[n-2])


    return NL_re, NL_im

# n = 44
# k0 = 0.125
# lmb = 2.0
# K = np.array([k0*lmb**i for i in range(22)],dtype=np.float64)
# nu = 1.0e-7
# # x_past = np.zeros((2,n))
# # x_past[0,0::2] = K**(-1./3) #[0:int(n/2)]
# # x_past[0,1::2] = 1e-4 
# x_p_r,x_p_i = NL(state[0,0::2],state[0,1::2])
# state[1,0::2] = np.exp(-nu*K**2*1.0e-5)*(state[0,0::2] + 1.0e-5*x_p_r ) #np.random.normal(0,1.e-4,size=((n,2))) # state at time t-1 and t-2 for the model m
# state[1,1::2] = np.exp(-nu*K**2*1.0e-5)*(state[0,0::2] + 1.0e-5*x_p_i ) #np.random.normal(0,1.e-4,size=((n,2))) # state at time t-1 and t-2 for the model m


# intégration sur n pas

for i in range(2,n_step):

    (Xp_out, Yp_out), (state[i,0::2],state[i,1::2]) = model.integrate(state[i-2,0::2], state[i-2,1::2],
                state[i-1,0::2], state[i-1,1::2],  # ← 1::2 pour Y
                n_steps=999)
ecart_x = state[:,0::2]-X_ref
ecart_y = state[:,1::2]-Y_ref

print("mean x   std x")
print(np.mean(ecart_x),np.std(ecart_x))

print("mean y   std y")
print(np.mean(ecart_y),np.std(ecart_y))

print("RMSE X")
print(np.sqrt(np.mean(ecart_x**2)))

print("RMSE Y")
print(np.sqrt(np.mean(ecart_y**2)))

# Xp_out/Yp_out  →  état à t + (n-1)*dt
# X_out/Y_out    →  état à t + n*dt
for i in range(10):
    plt.figure()
    plt.plot(X_ref[:,2*i])
    plt.plot(state[:,2*i])

    plt.figure()
    plt.plot(X_ref[:,2*i]-state[:,2*i])

plt.show()

plt.savefig("/home/s26calme/Documents/code_stage/KF/test/test")
