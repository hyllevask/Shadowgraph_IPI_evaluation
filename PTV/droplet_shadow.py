#mainfile for running

#Imort libreries
import argparse
import os
import numpy as np

#Setup the parser for commandline interface
parser = argparse.ArgumentParser(description='Process the shadowgraph images.')
parser.add_argument('--indir',type=str,default='data',help = 'Input Directory')
parser.add_argument('--crop',type=tuple,default=(220,530,220,530),help="Crop Limits")
parser.add_argument('--save_images',type=int, default=0,help = "Saves masks and histograms")
parser.add_argument('--pixelpitch', type=float,default=1,help="Pixel Pitch in the image")
parser.add_argument('--th', type=int,default=180,help='Threshold for binarization')
args = parser.parse_args()

#Setup the main function

def main():
    #Pandas is used to store the data for each frame and
    import pickle

    print("th: %i" % args.th)
    listan = []     #The numpy arrays will be stored in this array
    if args.save_images == 1:
        print("Image Save Enabled")
        if not os.path.exists(args.indir +"/result_images"):
            os.mkdir(args.indir +"/result_images")       #Make result folder if it does not exsist
    #Loop over the files in the dir
    for ii,filename in enumerate(os.listdir(args.indir)):
        print(ii)
        if filename.endswith('.bmp'):
            #Call the main analyzing function
            data = analyze_image(filename,ii,args.save_images)
            listan.append(data)
        else:
            continue
    save_path = args.indir + "\processed_data.p"
    pickle.dump(listan,open(save_path,'wb'))
    #Create a dictionary from the list of results and make a dataframe from it
    #df = pd.DataFrame({ i:pd.Series(value) for i, value in enumerate(listan) })
    #Save the dataframe as csv
    #df.to_csv('./df_list.csv')

def analyze_image(filename,ii,save_images):

    import matplotlib.pyplot as plt
    import skimage.measure
    import skimage.exposure

    #read and crop the input
    im = plt.imread(args.indir +'/'+ filename)
    crop_limits = args.crop
    #im = im[crop_limits[0]:crop_limits[1],crop_limits[2]:crop_limits[3]]

    #Do the thresholding
    im_bw = im < args.th
    #Label and get regionprops
    all_labels = skimage.measure.label(im_bw)
    props = skimage.measure.regionprops(all_labels)
    data = []
    #Extract the area for each particle
    for prop in props:
        r,c = prop.centroid
        data.append(np.array([r*args.pixelpitch,c*args.pixelpitch,np.sqrt(prop.area)*args.pixelpitch]))
    if save_images == 1:
        #Generate and save images
        plt.figure(1,clear=True)
        plt.imshow(im < args.th)
        plt.savefig(args.indir + '/result_images/bw'+str(ii))
        #plt.figure(2,clear=True)
        #hist, hist_centers =skimage.exposure.histogram(np.array(area))
        #plt.plot(hist_centers,hist,lw=2)
        #plt.savefig('result_images/hist'+str(ii))
    return data
if __name__ == "__main__":
    main()