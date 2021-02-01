import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pickle.load(open('/home/johan/Documents/Datasets/Measurements/Corona/Test_dataprocessed_IPI_data.p','rb'))

total_list = []
for ii,dd in enumerate(data):
	r_temp = [t[2] for t in dd]
	total_list = total_list + r_temp

size_data = np.array(total_list)*1e6	#Make µm
plt.hist(size_data,bins = 21,range=(0,25),edgecolor="k")
plt.xlabel(r"Particle Diameter $[ \mu m]$")
plt.ylabel("Count")
plt.savefig("Histogram_25",dpi=600)
