from Kalman_Tracker import Tracker
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.ion()

#Generate or load data

#Generate
def generate_data(filename,N_frames):
    import pickle
    size = 10
    start1 = np.array([50,0,10])
    start2 = np.array([0,0,20])
    data = [start1,start2]
    data_list = []
    for ii in np.arange(0,N_frames):
        data = [np.array([10,(jj+1),0]) + x for jj,x in enumerate(data)]
        data = [x + np.random.randn(3)*np.array([1,1,0.2]) for x in data]
        data_list.append(data)
    pickle.dump(data_list,open(filename, "wb"))

generate_data('test_data.p',100)
data_list = pickle.load(open('test_data.p','rb'))


tracker = Tracker()

for dd in data_list:
    tracker.Update_tracks(dd)
x_list = []
y_list = []
r_list = []
for ii,track in enumerate(tracker.tracks):
    x_list.append(np.array([a[0] for a in track.trace]))
    y_list.append(np.array([a[1] for a in track.trace]))
    r_list.append(np.array([a[4] for a in track.trace]))


f1 = plt.figure(1,clear=True)
a1 = f1.gca()
p1 = a1.plot(x_list[0],y_list[0])
plt.show()

f2 = plt.figure(2,clear=True)
a2 = f2.gca()
p2 =a2.hist(r_list[0])
plt.show()

f3 = plt.figure(3,clear=True)
a3 = f3.gca()
p3 = a3.plot(x_list[1],y_list[1])
plt.show()

f4 = plt.figure(4,clear=True)
a4 = f4.gca()
p4 = a4.hist(r_list[1])
plt.show()