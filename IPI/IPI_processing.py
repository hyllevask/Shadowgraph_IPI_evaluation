#mainfile for running

#Imort libreries
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure, skimage.exposure, skimage.filters
from skimage.feature import blob_log
import pandas as pd

#Setup the parser for commandline interface
parser = argparse.ArgumentParser(description='Process the IPI images.')
parser.add_argument('--indir',type=str,default='data',help = 'Input Directory')
parser.add_argument('--crop',type=tuple,default=(200,600,200,600),help="Crop Limits")
parser.add_argument('--save_images',type=int, default=0,help = "Saves masks and histograms")
args = parser.parse_args()

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
    #df = pd.DataFrame({ i:pd.Series(value) for i, value in enumerate(listan) })
    #Save the dataframe as csv
    #df.to_csv('./df_list.csv')

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
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    for blob in blobs:
        x, y, r = blob
        c = plt.Circle((y,x),r)
        ax.add_patch(c)
    plt.show()

    #todo add the sizing part from the fringe patterns



if __name__ == "__main__":
    main()
