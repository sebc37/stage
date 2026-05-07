import numpy as np
from scipy.integrate import odeint as scipy_odeint
import matplotlib.pyplot as plt
import glob

PATH = "/home/s26calme/Documents/code_stage/PINNs/causal_PINN/"

list_data = glob.glob(PATH + "*.npy")
data = []
for l in list_data:
    data.append(np.load(l,allow_pickle=True))

rho, sigma, beta = 28.0, 10.0, 8.0 / 3.0


def plot_attractor_(x,y,name):
    
    ax = plt.figure().add_subplot(projection='3d')
    # Unpack columns — works for any (T/dt, 3) numpy array
    ax.plot(x[0][:, 20], x[1][:, 20], x[2][:, 20],
        lw=0.5, alpha=0.85, color="royalblue", label="Prediction")

    ax.plot(y[:, 0], y[:, 1], y[:, 2],
        lw=0.5, alpha=0.85, color="tomato",    label="Truth")

    ax.set_title(f"Lorenz Attractor {name}")
    ax.legend()
    plt.savefig(PATH + f'Lorenz_Attractor_{name}')
    #plt.show()


def f(state, t):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

T = 30
t_full = np.arange(0.0, T, 0.01)
ref    = scipy_odeint(f, [1.0, 1.0, 1.0], t_full)

# c=0
# a = 1000000
# for i in range(50):
#     a = np.mean(data[0][:,i]-ref[:,0])
#     data[0][:,i] = data[0][:,i]-ref[:,0]
#     if data[0][:,i]-ref[0] <= a:
#         c=i

plt.figure()

plt.plot(data[0][:,-2])
plt.plot(data[1][:,-2])
plt.plot(data[3][:,-2])
plt.show()



plot_attractor_([data[0],data[1],data[3]],ref,'causal_pinn_dif')
