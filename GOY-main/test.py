import matplotlib
import matplotlib.pyplot as plt
import numpy
from scipy import signal
matplotlib.use("QtAgg")

## load results:
data=numpy.loadtxt("/home/s26calme/Documents/code_stage/GOY-main/data.dat")
x   =data[:,::2]
v   =numpy.loadtxt("v.dat" )

k0=0.125
N_shell=x.shape[1]
Npts=v.shape[0]
fs=100
H=1./3

modes=2**numpy.arange(N_shell)*k0


print("shell :   ", x.shape[0], "pts in time for", x.shape[1], "shells")
print("velocity :", v.shape[0], "pts in time")

S  = numpy.var(x, axis=0)
[fr, Sr] = signal.periodogram(v, fs=fs, window='hann')

## plot reconstructed velocity signal
Fig=plt.figure(figsize=(8,5))
P1 =Fig.add_subplot(1,1,1)
P1.plot(numpy.arange(Npts)/fs, v)
P1.set_ylabel(r"Velocity")

## plot order 2 statistics
Fig=plt.figure(figsize=(8,5))
P1 =Fig.add_subplot(1,1,1)
P1.set_xlabel(r"wavenumber")
P1.set_ylabel(r"order 2")

P1.plot(numpy.log(modes), numpy.log(S), '--o', label=r"shell")
P1.plot(numpy.log(fr), numpy.log(Sr), '--', label=r"$|\delta v^2|(k)$")
mo=numpy.array([-3, 13])
P1.plot(mo, -2*H*mo-5, 'k--', label=r"$-2/3$")
P1.legend()
plt.show()

