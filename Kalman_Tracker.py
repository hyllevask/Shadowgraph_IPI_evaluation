from filterpy.kalman import KalmanFilter
import numpy as np

class Track(object):
    def __init__(self,x,y,r,t_ID):
        self.kf = KalmanFilter(dim_x=5,dim_z=3)        #(x,y,dx,dy,r)
        self.kf.x = np.array([x,y,0,0,r])
        self.kf.F = np.array([[1,0,dt,0,0],
                         [0,1,0,dt,0],
                         [0,0,1,0,0],
                         [0,0,0,1,0],
                         [0,0,0,0,1]])

        self.kf.H = np.array([1,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,0,1])
        self.kf.P =* 100.
        self.kf.R = np.array([[10],
                         [10],
                         [40]])

        self.track_ID = t_ID
        self.undetected_frames = 0
        self.trace = []
        

    def update(self,x,y,r):
        z = np.array([x,y,r])
        self.kf.predict()
        self.kf.update(z)
class Tracker(object):


    def __init__(self):
