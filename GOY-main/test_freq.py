
N_fs= int(1/1e-5/100)
print(N_fs)
c = 0
for i in range(int(1000/1e-5)):
    if i%N_fs==0:
        print(i)
        print(i%N_fs)
        c+=1
print(c)