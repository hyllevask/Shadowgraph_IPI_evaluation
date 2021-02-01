import pickle
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
path = "C:/Users/johohm/Documents/Corona/Baseline/PulsedLaser/20201015/andning1/processed_data.p"

data = pickle.load(open(path,'rb'))
max_cost = 25
particles_in_frame = []
pp = 40e-3/2448
dt = 1/5
dt_double = 50e-6
#frame1 = [np.array([1500,1700,4]),np.array([1000,1500,4]),np.array([1100,1500,4]),np.array([1300,1700,4]),np.array([1200,1200,100])]
#frame2 = [x+10 for x in frame1 if not x[2] == 100]
#data = [frame1,frame2]
mean_dx = []
mean_dy = []
N_particles = []
for kk,frame in enumerate(data):
    all_x = np.array([])
    all_y = np.array([])
    all_dx = np.array([])
    all_dy = np.array([])
    print(frame.__len__())
    particles_in_frame.append(frame.__len__())
    x_data = []
    y_data = []
    for particle in frame:
        x_data.append(particle[0])
        y_data.append(particle[1])
    if kk%2 == 1:

        N = len(prev_x_data)
        M = len(x_data)
        all_detects = list(range(M))
        cost = np.zeros(shape=(N, M))
        print("Predict")
        for ii,(x_prev,y_prev) in enumerate(zip(prev_x_data,prev_y_data)):

            for jj,(x,y) in enumerate(zip(x_data,y_data)):


                distance = np.sqrt((x_prev - x) ** 2 + (y_prev - y) ** 2)
                if distance < max_cost:
                    cost[ii][jj] = distance
                else:
                    cost[ii][jj] = 1e99
        row_ind, col_ind = linear_sum_assignment(cost)
        dx = []
        dy = []
        x = []
        y = []
        for (a,b) in zip(row_ind,col_ind):
            dx_temp = x_data[b] - prev_x_data[a]
            dy_temp = y_data[b] - prev_y_data[a]
            if np.sqrt(dx_temp**2 + dy_temp**2) < max_cost:
                x.append(prev_x_data[a])
                y.append(prev_y_data[a])

                dx.append(dx_temp)
                dy.append(dy_temp)
        x = np.array(x)*pp
        y = np.array(y)*pp
        dx = np.array(dx)*pp/dt_double
        dy = np.array(dy)*pp/dt_double

        all_x = np.concatenate((all_x,x))
        all_y = np.concatenate((all_y, y))
        all_dx = np.concatenate((all_dx, dx))
        all_dy = np.concatenate((all_dy, dy))

        mean_dx.append(np.mean(all_dx))
        mean_dy.append(np.mean(all_dy))
        N_particles.append(all_x.size)

        if False:
            fname = 'test9/bild'+str(kk)
            fig,ax = plt.subplots()
            ax.scatter(x,y)
            ax.quiver(x,y,dx,dy,scale)
            plt.savefig(fname, dpi=600)
            plt.close(fig)

        #grid_x,grid_y = np.mgrid[0:1:2448j,0:1:2050j]
        #gg_x = griddata(np.vstack((all_x, all_y)).T, all_dx, (grid_x, grid_y), method='cubic')
        #gg_y = griddata(np.vstack((all_x, all_y)).T, all_dy, (grid_x, grid_y), method='cubic')

        #fname = 'test9/xgrid' + str(kk)
        #fig, ax = plt.subplots()
        #ax.imshow(gg_x)
        #plt.savefig(fname, dpi=600)
        #plt.close(fig)

        #fname = 'test9/ygrid' + str(kk)
        #fig, ax = plt.subplots()
        #ax.imshow(gg_y)
        #plt.savefig(fname, dpi=600)
        #plt.close(fig)

        #fname = 'test9/hist' + str(kk)
        #fig, ax = plt.subplots()
        #ax.hist(all_dx, alpha=0.5, bins=np.arange(-15, 15, 1))
        #ax.hist(all_dy, alpha=0.5, bins=np.arange(-15, 15, 1))
        #plt.savefig(fname, dpi=600)
        #plt.close(fig)

    #Assign the current data to be used as prev data in the next iteration.
    prev_x_data = x_data
    prev_y_data = y_data


t = np.arange(0,mean_dx.__len__(),1)*dt
fig,(ax1,ax2) = plt.subplots(nrows=2,sharex=True)
ax1.plot(t,N_particles)
ax1.title.set_text('Number of particles')
ax1.set_ylabel('Count')

ax2.plot(t,mean_dx)
ax2.plot(t,mean_dy)
ax2.title.set_text('Mean Particle Velocity')
ax2.set_xlabel('time [s]')
ax2.set_ylabel('Velocity [m/s]')
ax2.legend(('$v_x$','$v_y$'))