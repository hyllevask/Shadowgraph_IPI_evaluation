from Kalman_Tracker import Tracker
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
data0 = [np.array([10,10,10]),np.array([50,50,5])]
tracker = Tracker()

for ii in np.arange(0,100):
    print(ii)

    data = [np.array([ii,ii,1])*x + np.random.randn(3) for x in data0]
    tracker.Update_tracks(data)
x_list = []
y_list = []
r_list = []
for ii,track in enumerate(tracker.tracks):
    x_list.append(np.array([a[0] for a in track.trace]))
    y_list.append(np.array([a[1] for a in track.trace]))
    r_list.append(np.array([a[2] for a in track.trace]))

plt.plot(x_list[0],y_list[0])
plt.show()

plt.hist(r_list[0])
plt.show()