#mainfile for running

#Imort libreries
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure, skimage.exposure, skimage.filters
from skimage.feature import blob_log
import pandas as pd
from scipy.signal import correlate2d, find_peaks


#Setup the parser for commandline interface
parser = argparse.ArgumentParser(description='Process the IPI images.')
parser.add_argument('--indir',type=str,default='data',help = 'Input Directory')
parser.add_argument('--crop',type=tuple,default=(200,600,200,600),help="Crop Limits")
parser.add_argument('--save_images',type=int, default=0,help = "Saves masks and histograms")
args = parser.parse_args()

prop = {'m':1.33, 'lamb':532e-9, 'theta':np.pi/2, 'f_num':4}


def main():
    #Pandas is used to store the data for each frame and

    listan = []     #The numpy arrays will be stored in this array
    if args.save_images == 1:
        print("Image Save Enabled")
        if not os.path.exists("IPI_result_images"):
            os.mkdir("result_images")       #Make result folder if it does not exsist
    #Loop over the files in the dir
    for ii,filename in enumerate(os.listdir(args.indir)):
        print(ii)
        if filename.endswith(('.bmp','.png')):
            #Call the main analyzing function
            area = analyze_IPI(filename,ii,args.save_images)
            listan.append(area)
        else:
            continue
    #Create a dictionary from the list of results and make a dataframe from it
    df = pd.DataFrame({ i:pd.Series(value) for i, value in enumerate(listan) })
    #Save the dataframe as csv
    df.to_csv('./df_list.csv')

def analyze_IPI(filename,ii,save_images):
    #Finds the defocused particles using LoG blob detection.
    #Each particle is then processed to find the size estimate

    #Read images
    im = plt.imread(args.indir + '/' + filename)
    #Use the LoG blob detection
    blobs = blob_log(im,min_sigma=8,max_sigma=20,num_sigma=10)
    #Scale to correct radii
    blobs[:,2] = blobs[:,2]*np.sqrt(2)

    #plot
    #todo make it optional to plot
    #fig,ax = plt.subplots(1)

    #ax.imshow(im, cmap='gray')
    area = []
    for ii,blob in enumerate(blobs):
        x, y, r = blob
        r_rounded = np.floor(r)
        subimage = im[int(x-r_rounded):int(x+r_rounded),int(y-r_rounded):int(y+r_rounded)]
        #plt.imshow(subimage)
        #plt.show()
        shift = analyze_fringes(subimage)
        if shift == -1:
            continue
        N_fringes = 2*r/shift
        size = fringes2size(N_fringes, prop['m'], prop['lamb'], prop['f_num'],prop['theta'])
        print('Particle %i: %i fringes, %f um' % (ii,N_fringes,size*1e6))
        area.append(size)
    return np.array(area)


    #todo add the sizing part from the fringe patterns

def analyze_fringes(subimage):
    im_norm = (subimage - np.mean(subimage))
    #todo prehaps implement a fringe contrast
    #fringe_contrast = (np.max(subimage[5:25,5:25]) - np.min(subimage[5:25,5:25]))/(np.max(subimage[5:25,5:25]) + np.min(subimage[5:25,5:25]))
    #print("Fringe Contrast: %f" % fringe_contrast)
    test = correlate2d(im_norm,im_norm,mode='same')
    test = test[15:,15]
#   For dedugging
#    lags = np.arange(0,17)
#    test2 = test /(16-lags + 1)
#    test2 = test2/test2[0]
#    test = test/test[0]
#    plt.plot(test)
#    plt.plot(test2)
#    plt.show()
    peaks = find_peaks(test)

    sorted_peaks = np.sort(peaks[0])
    if sorted_peaks.__len__() != 0:
        first_peak = sorted_peaks[0]
    else:
        first_peak = -1

    return first_peak

def fringes2size(N,m,lamb,f_num,theta):
    alfa = np.arcsin(1/f_num/2/2)

    A = 2*lamb/alfa
    B = (np.cos(theta/2) + (m*np.sin(theta/2)) / np.sqrt(m**2-2*m*np.cos(theta/2) + 1))

    d = N*A/B
    return d


if __name__ == "__main__":
    main()
