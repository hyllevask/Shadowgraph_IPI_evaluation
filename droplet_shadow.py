#mainfile for running
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process the shadowgraph images.')
parser.add_argument('--indir',type=str,default='data',help = 'Input Directory')
parser.add_argument('--crop',type=tuple,default=(200,600,200,600),help="Crop Limits")
parser.add_argument('--save_images',type=int, default=0,help = "Saves masks and histograms")
args = parser.parse_args()

def main():

    import pandas as pd
    df = pd.DataFrame(columns=['Particle','Area'])
    index = 0
    listan = []
    if args.save_images == 1:
        print("apa")
        if not os.path.exists("result_images"):
            os.mkdir("result_images")
    for ii,filename in enumerate(os.listdir(args.indir)):
        print(ii)
        if filename.endswith('.bmp'):

            area = analyze_image(filename,ii,args.save_images)
            listan.append(area)
        else:
            continue
    df = pd.DataFrame({ i:pd.Series(value) for i, value in enumerate(listan) })
    df.to_csv('./df_list.csv')

def analyze_image(filename,ii,save_images):
    import skimage
    import matplotlib.pyplot as plt
    import skimage.measure
    import skimage.exposure
    im = plt.imread(args.indir +'/'+ filename)
    crop_limits = args.crop
    im = im[crop_limits[0]:crop_limits[1],crop_limits[2]:crop_limits[3]]
    im_bw = im < 180
    all_labels = skimage.measure.label(im_bw)
    props = skimage.measure.regionprops(all_labels)
    area = []
    for prop in props:
        area.append(prop.area)
    if save_images == 1:
        print("hej")
        plt.figure(1,clear=True)
        plt.imshow(im < 180)
        plt.savefig('result_images/bw'+str(ii))
        plt.figure(2,clear=True)
        hist, hist_centers =skimage.exposure.histogram(np.array(area))
        plt.plot(hist_centers,hist,lw=2)
        plt.savefig('result_images/hist'+str(ii))
    return np.array(area)
if __name__ == "__main__":
    main()