import cv2
import numpy as np

# get  mean and std of depth image data 
# based on a sample of the images, just to keep the code simple
p = 0.25

if __name__ == "__main__":
    with open("data/training.txt") as f:
        all_hha = np.zeros((1, 480, 640, 3))
        for i,line in enumerate(f):
            if np.random.rand(1).item() > p:
                continue
            print("reading HHA image " + str(i))
            rgb, depth, label = line.strip().split('\t')
            hha = cv2.imread(depth)
            all_hha = np.append(all_hha, np.expand_dims(hha, axis=0), axis=0)
        average_hha = all_hha.mean(axis=(0, 1, 2))
        std_hha = all_hha.std(axis=(0, 1, 2))
        print("on a 255 scale:")
        print("HHA mean: " + str(average_hha))
        print("HHA std: " + str(std_hha))
        all_hha = all_hha / 255
        average_hha = all_hha.mean(axis=(0, 1, 2))
        std_hha = all_hha.std(axis=(0, 1, 2))
        print("on a 1 scale:")
        print("HHA mean: " + str(average_hha))
        print("HHA std: " + str(std_hha))
        
