import time
import numpy as np
import matplotlib.pyplot as plt


PATH = r"./Donnees/"
path_data = PATH + "data.dat"

data = np.loadtxt(path_data)

X = data[:,::2]

def time_series(X,X_f,Std_mode,perc,mean):
    time = np.linspace(0,np.shape(X)[0],np.shape(X)[0])
    plt.figure(1)

    plt.plot(time,(X[:,3]-mean[3])/Std_mode[3], 'black',label='Mode 4 Forcing',alpha=1)
    plt.plot(time,(X[:,5]- mean[5])/Std_mode[5], 'gray',label='Mode 6',alpha=0.5)
    plt.plot(time,(X[:,7]-mean[7])/Std_mode[7], 'gray',label='Mode 8',alpha=0.5)
    plt.plot(time,(X[:,9]-mean[9])/Std_mode[9], 'gray',label='Mode 10',alpha=0.5)
    #plt.plot(time,X[:,21]/Std_mode[21], 'b',label='Mode 21',alpha=1)
   
    plt.plot(time,(X_f[:,5]-mean[5])/Std_mode[5], '+',label='Mode 6 filtered')
    plt.plot(time,(X_f[:,7]-mean[7])/Std_mode[7], '+',label='Mode 8 filtered')
    plt.plot(time,(X_f[:,9]-mean[9])/Std_mode[9], '+',label='Mode 10 filtered')
    plt.xlabel('Time')
    plt.ylabel('Velocities')
    plt.legend(title=f'Percentage filtered: {100-perc:.2f}%')
    plt.show()

def plot_mode_reduit(Var_mode):

    mode = [i for i in range(len(Var_mode))]
    #log_var = np.log2(Var_mode)
    plt.semilogy(mode,Var_mode,'+',label='Log Variance modes')
    plt.xlabel('Mode k')
    plt.ylabel(r"$E(k)$")
    plt.show()


def filter_mode(X,mode_min:int ,ratio:float,seed):
    
    np.random.seed(seed=seed)
    X_subset = np.copy(X)
    X_subset[:,0:mode_min] = None #enlève les modes inférieurs à mode_min
    X_subset[0:20000,:] = None #enlève la phase de stabilisation
    nb_column = np.shape(X_subset)[1]
    nb_line = np.shape(X_subset)[0]
    X_filtered = np.copy(X_subset)
    for j in range(mode_min,nb_column):
        for i in range(nb_line):
            if (np.random.random()<=ratio):
                X_filtered[i,j] = X_subset[i,j] 
            else:
                X_filtered[i,j] = None
    var_mode = [np.var(X[:,j]) for j in range(nb_column)]
    std_mode = [np.std(X[:,j]) for j in range(nb_column)]
    mean_mode = [np.mean(X[:,j]) for j in range(nb_column)]
    pourcentage_filtered = nb_line*ratio/nb_line*100
    return X_filtered,var_mode,std_mode,pourcentage_filtered,mean_mode

X_f ,Var_mode,Std_mode,perc,mean = filter_mode(X,5,0.001,123456)
X_mask = np.isnan(X_f)
#print(np.shape(X_f))
time_series(X=X,X_f=X_f,Std_mode=Std_mode,perc=perc,mean=mean)
plot_mode_reduit(Var_mode=Var_mode)