import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#data = pickle.load(open('processed_IPI_data_run7.p','rb'))
data = pickle.load(open('/home/johan/Documents/processed_IPI_data.p','rb'))

total_list = []
N = []
r_mean = []
for ii,dd in enumerate(data):
	r_temp = [t[2] for t in dd if t[2] > 2]
	total_list = total_list + r_temp

	r_mean.append(np.mean(np.array(r_temp)))
	N.append(r_temp.__len__())

	test = [d[2] for d in dd]

	#plt.figure()
	#plt.plot(test)
	#plt.show()

size_data = np.array(total_list)	#Make µm
plt.hist(size_data,bins = range(0,20,1),edgecolor="k")
plt.xlabel(r"Particle Diameter $[ \mu m]$")
plt.ylabel("Count")
plt.savefig("new/Histogram_run16",dpi=600)
plt.close()


plt.plot(np.arange(0,ii+1),np.array(r_mean))
plt.ylim((0,100))
plt.xlabel("Frame #")
plt.ylabel(r"Mean Particle Diameter $[ \mu m]$")
plt.savefig("new/Mean_size",dpi=600)
plt.close()


plt.plot(np.arange(0,ii+1),np.array(N))
plt.xlabel("Frame #")
plt.ylabel("Count")
plt.savefig("new/N_particles",dpi=600)