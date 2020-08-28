from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment
# Partly inspired by https://github.com/srianant/kalman_filter_multi_object_tracking/blob/master/tracker.py

class Track(object):
    """"
    Class for each track. A kalman filter estimates the position, velocity and size of each particle

    """
    def __init__(self,x,y,r,t_ID,dt):
        self.kf = KalmanFilter(dim_x=5,dim_z=3)        #(x,y,dx,dy,r)
        self.kf.x = np.array([x,y,0,0,r])
        self.kf.F = np.array([[1,0,dt,0,0],
                         [0,1,0,dt,0],
                         [0,0,1,0,0],
                         [0,0,0,1,0],
                         [0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,0,1]])
        self.kf.P = self.kf.P * 100.
        #self.kf.R = np.array([[10],
        #                 [10],
        #                 [40]])

        self.track_ID = t_ID
        self.undetected_frames = 0
        self.trace = []


    def update(self,x,y,r):
        z = np.array([x,y,r])
        self.kf.predict()
        self.kf.update(z)
        self.trace.append(self.kf.x)
class Tracker(object):


    def __init__(self):
        self.tracks = []
        self.deleted_tracks = []
        self.track_count = 0
        self.dt = 1

    def Update_tracks(self,data):
        #Data is a list where each element is a tuple with x,y,r

        #If no tracks are stored create new tracks from the data
        if (len(self.tracks) == 0):
            for d in data:
                x,y,r = d
                temp_track = Track(x,y,r,self.track_count,self.dt)
                self.tracks.append(temp_track)


        # DATA ASSOCIATION

        N = len(self.tracks)
        M = len(data)
        cost = np.zeros(shape=(N, M))
        for ii,track in enumerate(self.tracks):
            #Predicts every excisting track
            track.kf.predict()
            for jj, d in enumerate(data):
                x,y,r = d
                x_pred = track.kf.x[0]
                y_pred = track.kf.x[1]
                r_pred = track.kf.x[-1]

                distance = np.sqrt((x_pred-x)**2 + (y_pred-y)**2 + (r_pred-r)**2)
                cost[ii][jj] = distance

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            #Flag each element that will be used
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        #Check for unassigned tracks

        #CHeck is track should be deleted
        for ii,track in enumerate(self.tracks):
            if track.undetected_frames > 2:
                self.tracks[ii] = -1

        new_deleted = [track for track in self.tracks if track.undetected_frames > 3 ]
        new_tracks = [track for track in self.tracks if track.undetected_frames <= 3 ]

        self.tracks = new_tracks
        self.deleted_tracks.append(new_deleted)
        #CHeck for unassigned detects

        #Possibly start new tracks


        #Loop over the tracks and update the state with the assigned measurement
        #THe index off assignment is the track and the vale is the corresponding data
        for ii,ass in enumerate(assignment):
            self.tracks[ii].kf.update(data[ass][:,np.newaxis])
            self.tracks[ii].trace.append(self.tracks[ii].kf.x)


