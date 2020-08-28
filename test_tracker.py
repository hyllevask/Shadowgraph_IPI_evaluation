from Kalman_Tracker import Tracker
import numpy as np
data = [np.array([10,10,10]),np.array([50,50,5])]
tracker = Tracker()
tracker.Update_tracks(data)


#data2 = 