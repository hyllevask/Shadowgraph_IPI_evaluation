from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment
# Partly inspired by https://github.com/srianant/kalman_filter_multi_object_tracking/blob/master/tracker.py

#todo FIxa så att man kan mata in de förväntade brusnivåerna in i trackern

class Track(object):
    """"
    Class for each track. A kalman filter estimates the position, velocity and size of each particle

    """
    def __init__(self,x,y,r,t_ID,dt,Q_noise, R_noise):
        self.kf = KalmanFilter(dim_x=5,dim_z=3)        #(x,y,dx,dy,r)
        self.kf.x = np.array([x,y,0,0,r])
        self.kf.F = np.array([[1,0,dt,0,0], #Transition model
                         [0,1,0,dt,0],
                         [0,0,1,0,0],
                         [0,0,0,1,0],
                         [0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0],  #Observation model
                        [0,1,0,0,0],
                        [0,0,0,0,1]])
        self.kf.P = self.kf.P * 100.    #Initial covariance matrix
        self.kf.R = np.diag(R_noise)         #Measurement noise
        self.kf.Q = np.diag(Q_noise)         #Process noise

        self.track_ID = t_ID
        self.undetected_frames = 0
        self.life = 1
        self.trace = []
        self.tentative = True

    def update(self,x,y,r):
        z = np.array([x,y,r])
        self.kf.predict()
        self.kf.update(z)
        self.trace.append(self.kf.x)
class Tracker(object):


    def __init__(self,dt,R,Q):
        self.tracks = []
        self.deleted_tracks = []
        self.tentative_tracks = []
        self.track_count = 0
        self.dt = dt
        self.max_cost = 3000
        self.defult_Q = Q


        #self.defult_P =
        self.defult_R = R

    def Update_tracks(self,data,dt = 0):
        #Data is a list where each element is a tuple with x,y,r
        #It is possible to add a custum timestep (if the sampling is uneven). Defult dt is taken from the creation
        #If no tracks are stored create new tracks from the data

        if dt == 0:
            dt = self.dt

        if (len(self.tracks) == 0):
            print("Initiating Tracks")
            for d in data:
                x,y,r = d
                temp_track = Track(x,y,r,self.track_count,self.dt,self.defult_Q,self.defult_R)
                temp_track.trace.append(temp_track.kf.x)
                self.tracks.append(temp_track)
                self.track_count += 1
            print("Complete")
            return
        #print("Updating Tracks")
        # DATA ASSOCIATION

        N = len(self.tracks)
        M = len(data)
        all_detects = list(range(M))
        cost = np.zeros(shape=(N, M))
        print("Predict")
        for ii,track in enumerate(self.tracks):
            #Predicts every excisting track
            x_pred = track.kf.x[0]
            y_pred = track.kf.x[1]
            r_pred = track.kf.x[-1]
            track.kf.predict()
            track.life += 1
            for jj, d in enumerate(data):
                x,y,r = d


                distance = np.sqrt((x_pred-x)**2 + (y_pred-y)**2) #+ (r_pred-r)**2)
                if distance < self.max_cost:
                    cost[ii][jj] = distance
                else:
                    cost[ii][jj] = 1e99
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        print("assign")
        assignment = []
        for _ in range(N):
            #Flag each element that will be used
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            #print(cost[row_ind[i], col_ind[i]].sum())
            if cost[row_ind[i], col_ind[i]] == 1e99:
                assignment[row_ind[i]] = -2
                #print("hejhej")
            else:
                assignment[row_ind[i]] = col_ind[i]
                all_detects.remove(col_ind[i])
        print("update")
        #Loop over the tracks and update the state with the assigned measurement
        #THe index off assignment is the track and the vale is the corresponding data
        for ii,ass in enumerate(assignment):
            if ass != -2 and ass != -1:
                self.tracks[ii].undetected_frames = 0
                self.tracks[ii].kf.update(data[ass][:,np.newaxis])
                self.tracks[ii].trace.append(self.tracks[ii].kf.x)
                self.tracks[ii].life += 1
            else:
                #print("tjolahop")
                continue


        print("deleting")
        #Check for unassigned tracks
        for ii,track in enumerate(self.tracks):
            if assignment[ii] == -1 or assignment[ii] == -2:
                track.undetected_frames += 1
        #CHeck is track should be deleted
        #print("Deleting Track")
        for ii,track in enumerate(self.tracks):
            if track.undetected_frames > 2:
                if not track.tentative:
                    self.deleted_tracks.append(track)
                self.tracks[ii] = -1



        updated_tracks = [track for track in self.tracks if track != -1]

        self.tracks = updated_tracks
        print("creating")
        #CHeck for unassigned detects
        for detects in all_detects:
            d = data[detects]
            x,y,r = data[detects]
            self.tracks.append(Track(x,y,r,self.track_count,self.dt,self.defult_Q,self.defult_R))
            self.track_count += 1

        #print("Adding Track")
        #Possibly start new tracks
        for track in self.tracks:
            if track.life > 5 and track.tentative == True and track.undetected_frames < 3:

                track.tentative = False




    def get_results(self):
        #Returns the results for each track

        all_tracks = self.tracks + self.deleted_tracks
        results = np.zeros(shape=(all_tracks.__len__(), 3))
        for ii,track in enumerate(all_tracks):
            #print(ii)
            if track.trace.__len__() > 15 and not track.tentative:
                results[ii,0] = np.mean(np.array([x[-1] for x in track.trace]))
                results[ii,1] = np.mean(np.array([x[-3] for x in track.trace]))
                results[ii,2] = np.mean(np.array([x[-2] for x in track.trace]))
            else:
                results[ii,:] = np.nan
        results = results[~np.isnan(results).any(axis=1)]

        return results
