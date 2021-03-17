#mainfile for running

#Imort libreries
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure, skimage.exposure, skimage.filters
from skimage.feature import blob_log,blob_doh
from skimage.filters import sobel_v,gaussian
from scipy.signal import correlate2d
import pandas as pd
from scipy.signal import correlate2d, find_peaks
import pickle
from skimage.feature import peak_local_max

#Specify the arguments!



## FLAGS ##
save_images = 0
multi = 1
#indir = '/home/johan/Documents/Datasets/Measurements/Corona/Test_data'
indir = '/home/johan/Documents/run16'

#################
#Define properties for this data
prop = {'m':1.33, 'lamb':532e-9, 'theta':np.pi/2, 'f_num':4,'pp':50e-3/1080}

#TODO Use Pythorch for xcorr2



###################################################################################
#                           MAIN                                                  #
###################################################################################
def main():
    listan = []     #The numpy arrays will be stored in this array
    if save_images == 1:   
        print("Image Save Enabled")
        if not os.path.exists(indir +"/result_images"):
            os.mkdir(indir +"/result_images")                  #Make result folder if it does not exsist
   
    #Get list of files and sort them so they are in order (originally sorted by OS indexing)
    item_list = os.listdir(indir)
    item_list.sort()


    if multi == 1:
        filtered_list = [item for item in item_list if item.endswith(('.bmp','.png'))]
        print(filtered_list)
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        print("Running on % i threds" % num_cores)
        pool = multiprocessing.Pool(num_cores)

        #out1, out2, out3 = zip(*pool.map(analyze_IPI, item_list))
        data = pool.map(analyze_IPI, filtered_list)
        print(data)
        pickle.dump(data,open(indir +'/processed_IPI_data.p','wb'))
    else:
        print("Running on single thred")
        for ii,filename in enumerate(item_list):
            print(ii)
            #if ii == 10:
            #    break#For testing
            if filename.endswith(('.bmp','.png')):
                #Call the main analyzing function
                data = analyze_IPI(filename)
                listan.append(data)
            else:
                continue
        pickle.dump(listan,open(indir +'/processed_IPI_data.p','wb'))






###########################################################################
#                       FUNCTIONS                                         #
###########################################################################
def analyze_IPI(filename):
    print(filename)
    #Finds the defocused particles using LoG blob detection.
    #Each particle is then processed to find the size estimate
    print("Starting")

    im = plt.imread(indir + '/' + filename)
    #Use the LoG blob detection
   
    print("Correlating")
    particles = find_particles(im,63,3)
    r = 63

    if save_images == 1:
        print("Saving")
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        for particle in particles:
            y, x = particle
            r = 63
            c = plt.Circle((x, y), r,color='red', linewidth=2, fill=False)
            ax.add_patch(c)
        plt.savefig(indir +"/result_images/im"+filename[-3:]+"png",dpi = 600)
        plt.close(fig)

    #Hit fungerar det idag
    #TODO Se till att de 
    data = []
    for ii,particle in enumerate(particles):
        y, x = particle
        r_rounded = np.floor(r)
        subimage = im[int(x-r_rounded):int(x+r_rounded),int(y-r_rounded):int(y+r_rounded)]
        #plt.imshow(subimage)
        #plt.show()
        if subimage.size == 0:
            continue
        N_fringes = analyze_fringes_FFT(subimage,r_rounded)
        #if shift == -1:
        #    continue
        #N_fringes = 2*r/shift      #OLD FORMULATIONS


        size = fringes2size(N_fringes, prop['m'], prop['lamb'], prop['f_num'],prop['theta'])
        #print('Particle %i: %i fringes, %f um' % (ii,N_fringes,size*1e6))
        data.append(np.array([x*prop['pp'],y*prop['pp'],size]))

    print("Done")
    print(data)
    return data

def find_particles(im,d,s):
    im2 = rescale_im(im,20,200)
    grad_im = (sobel_v(gaussian(im2,sigma=s)))
    space = 10
    mask_im = np.zeros((d+space,d+space))     #r even
    

    x = np.arange(start=0,stop=d+space,step=1)

    grid = np.meshgrid(x,x)
    
    mask_im[(grid[0]-(d/2+space/2))**2 + (grid[1]-(d/2+space/2))**2 <(d/2)**2] = 1

    grad_mask = (sobel_v(gaussian(mask_im,sigma=s)))

    corr_map = correlate2d(grad_im,grad_mask,mode="full")
    c=corr_map
    c[c<2000] = 0

    coordinates = peak_local_max(c, min_distance=40) - (d+space)/2
    #print("Coordinates:")
    #print((coordinates[:,0]))
    return list(zip(coordinates[:,0],coordinates[:,1]))

def rescale_im(im,old_th,new_th):
    im = im/1
    #im[im<old_th]=0
    im = (im-old_th)/(255-old_th)
    im = im*(255-new_th)/(255-old_th)*(255-old_th) +new_th
    im[im<new_th] = 0
    return im


def analyze_fringes(subimage,r):
    r = int(r)
    im_norm = (subimage - np.mean(subimage))
    #todo prehaps implement a fringe contrast
    #fringe_contrast = (np.max(subimage[5:25,5:25]) - np.min(subimage[5:25,5:25]))/(np.max(subimage[5:25,5:25]) + np.min(subimage[5:25,5:25]))
    #print("Fringe Contrast: %f" % fringe_contrast)
    #print("r= %f" % r)
    test = correlate2d(im_norm,im_norm,mode='same')
    test = test[r-1:,r-1]
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


def analyze_fringes_FFT(subimage,r):
    #im_norm = (subimage - np.mean(subimage))
    im_norm = (subimage - np.min(subimage))/np.max(subimage)    #Map to [0,1]
    
    signal = im_norm.sum(axis=1)
    signal -= np.mean(signal)
    signal = np.pad(signal,int((256-signal.size)/2))
    signal_fft = np.fft.fft(signal)
    fft_fre=np.fft.fftfreq(n=signal.size,d=1)   #1/pixel

    #Make onesided
    signal_fft = np.abs(signal_fft[1:int(signal.size/2)])
    fft_fre = fft_fre[1:int(signal.size/2)]

    freq = fft_fre[np.argmax(signal_fft)]
    N = 2*r*freq

    #plt.plot(fft_fre,signal_fft)
    #plt.show()
    return N
def fringes2size(N,m,lamb,f_num,theta):
    alfa = 2*np.arcsin(1/f_num/2)

    A = 2*lamb/alfa
    B = (np.cos(theta/2) + (m*np.sin(theta/2)) / np.sqrt(m**2-2*m*np.cos(theta/2) + 1))

    d = N*A/B
    return d


if __name__ == "__main__":
    main()
