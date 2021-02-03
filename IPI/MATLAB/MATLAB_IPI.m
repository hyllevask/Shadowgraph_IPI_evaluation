
args.save_images = 1
args.indir = '/home/johan/Documents/Datasets/Measurements/Corona/Test_data'

args.pixelpitch = 50e-3/1080


m= 1.33;
lambda = 532e-9;
theta = pi/2;
f_num = 4;
pp = 50e-3/1080;




function [] = main(indir)
def main():
    listan = {}     %The numpy arrays will be stored in this cell
    if save_images == 1:   
        print("Image Save Enabled")
        if ~exist([indir,"result_images"], 'dir')
            mkdir([indir,"result_images"])
        end                %Make result folder if it does not exsist
    end
    %Get list of files and sort them so they are in order (originally sorted by OS indexing)
    item_list = dir(indir);
   
 
    for ii = 1:length(item_list)
        filename = itemlist.name(ii);
        if filename(end-3:end) == ".bmp"
            data = analyze_IPI(filename,ii,save_images)
            listan.append(data)
        else
            continue
        end

    %IMPLEMENT SAVE
    end
    end
    
function [data] = analyze_IPI(filename,ii,save_images)
    %Finds the defocused particles using LoG blob detection.
    %Each particle is then processed to find the size estimate


    im = imread(filename);
    %Use the LoG blob detection
   

    particles = find_particles(im,63,3);    %WHAT ARE THESE PARAMETERS?
    


    %blobs = blob_log(im,min_sigma=20,max_sigma=40,num_sigma=10,threshold=args.threshold)
    %Scale to correct radii
    %blobs[:,2] = blobs[:,2]*np.sqrt(2)

    %plot/save with circle
    if save_images == 1:
        print("Saving")
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        for particle in particles:
            y, x = particle
            r = 63
            c = plt.Circle((x, y), r,color='red', linewidth=2, fill=False)
            ax.add_patch(c)
        plt.savefig(args.indir +"/result_images/im"+str(ii),dpi = 600)
        plt.close(fig)


    data = []
    for ii = 1:length(particles)
        
        particle = particles(ii);
        % HIT ÄR JAG KLAR!
        
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
        data.append(np.array([x*args.pixelpitch,y*args.pixelpitch,size]))
    return data
    
    
    
    end