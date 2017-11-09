from embodied_ising import ising
import numpy as np
from sys import argv

# if len(argv) < 7:
# 	print("Usage: " + argv[0] + " <network size>" + " <number of sensors>" + " <number of motors>"  + " <Simulation time>"  " <Number of iterations>"  + " <Number of repetitions>" )
# 	exit(1)
#
# size=int(argv[1])
# Nsensors=int(argv[2])
# Nmotors=int(argv[3])
# T=int(argv[4])
# Iterations=int(argv[5])
# repetitions=int(argv[6])
size = 12
Nsensors = 6	# sensors[0,1,2] = position x binarized
			    # sensors[3,4,5] = position y binarized

Nmotors = 2     # potencial actions: np.linspace(-2,2,2**Nmotors)

T = 5000
Iterations = 1000
repetitions = 10

filename='correlations-ising2D-size400.npy'
Cdist=np.load(filename)

# repetitions = number of agents
for rep in range(repetitions):
	I=ising(size,Nsensors,Nmotors)
	I.m1=np.zeros(size)
	I.C1=np.zeros((size,size))
	for i in range(size):
		for j in range(max(i+1,Nsensors),size):
			ind=np.random.randint(len(Cdist))
			I.C1[i,j]=Cdist[ind]

	I.CriticalLearning(Iterations,T)


	filename='files/network-size_'+str(size)+'-sensors_'+str(Nsensors)+'-motors_'+str(Nmotors)+'-T_'+str(T)+'-Iterations_'+str(Iterations)+'-ind_'+str(rep)+'.npz'
	np.savez(filename, J=I.J, h=I.h,m1=I.m1,C1=I.C1)
