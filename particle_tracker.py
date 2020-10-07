from Kalman_Tracker import Tracker
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.ion()

#Generate or load data
#generate_data('test_data.p',100)
data_list = pickle.load(open('E:/Johan/Baseline/talking_p_2mm_fukt_C001H001S0001/processed_data.p','rb'))
#data_list = pickle.load(open('IPI/processed_IPI_data.p','rb'))
dt = 1/10000
print ('dt = %f' % dt)


R = np. array([50,50,50])  #Measurement noise
Q = np.array([100,100,1e15,1e15,1]) #Process noise

#Use the noise and timestep to create the tracker objekt
tracker = Tracker(dt,R,Q)
all_data_x = []
all_data_y = []

#For each timestep update the tracks
for ii,dd in enumerate(data_list):
    print(ii)
    x_temp = [t[0] for t in dd]
    y_temp = [t[1] for t in dd]
    all_data_x.extend(x_temp)
    all_data_y.extend(y_temp)
    tracker.Update_tracks(dd)
    #if ii == 30:
    #    break
#Remove the tentative tracks

x_list = []
y_list = []
r_list = []
for ii,track in enumerate(tracker.tracks):
    x_list.append(np.array([a[0] for a in track.trace]))
    y_list.append(np.array([a[1] for a in track.trace]))
    r_list.append(np.array([a[4] for a in track.trace]))

xd_list = []
yd_list = []
rd_list = []
for ii,track in enumerate(tracker.deleted_tracks):
    xd_list.append(np.array([a[0] for a in track.trace if not track.tentative]))
    yd_list.append(np.array([a[1] for a in track.trace if not track.tentative]))
    rd_list.append(np.array([a[4] for a in track.trace if not track.tentative]))

results = tracker.get_results()


f1 = plt.figure(1,clear=True)
a1 = f1.gca()
#for (x,y) in zip(x_list,y_list):
#    a1.plot(y,x)
#plt.draw()

#f2 = plt.figure(2,clear=True)
#a2 = f2.gca()
for (x,y) in zip(xd_list,yd_list):
    a1.plot(x,y)
plt.draw()
from scipy.stats import gaussian_kde
xs = np.linspace(0,300,200)
f3 = plt.figure(3,clear=True)
a3 = f3.gca()
a3.plot(gaussian_kde(results[:,0])(xs))
#a3.hist(results[:,0], bins=11)
plt.title("Size distribution")
plt.xlabel('Diameter [mm]')
plt.ylabel('Count')
plt.draw()

f4 = plt.figure(4,clear=True)
a4 = f4.gca()
a4.hist(np.sqrt(results[:,1]**2 + results[:,2]**2)/1e6, bins=21)
plt.title("Velocity distribution")
plt.xlabel('m/s')
plt.ylabel('Count')
plt.draw()


input("Press Any Key to Quit")
'''
f2 = plt.figure(2,clear=True)
a2 = f2.gca()
for frame in data_list:
    for particle in frame:
        a2.plot(particle[0],particle[1],'x')
plt.draw()

#f3 = plt.figure(3,clear=True)
#a3 = f3.gca()
#p3 = a3.plot


f3 = plt.figure(3,clear=True)
a3 = f3.gca()
p3 = a3.plot(x_list[1],y_list[1])
plt.show()

f4 = plt.figure(4,clear=True)
a4 = f4.gca()
p4 = a4.hist(r_list[1])
plt.show()
'''

plt.scatter(all_data_x,all_data_y)
plt.show()