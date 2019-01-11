'''
Created on 29.12.2018

@author: Nicco
'''

import csv
import sklearn
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import cv2
import random
import time


class data_loader():
    """
    this class loades all the data needed to train the model
    """
    def __init__(self, path):
        self.path = path
        self.csv_file=[]
        self.active_cams = {"left":True, "center":True, "right":True}
        self.training_samples = None
        self.validation_samples = None
        self.src = np.float32([[130, 65],[180, 65],
                          [319, 159],[0, 159]])
        self.dst = np.float32([[0, 0], [320, 0], 
                     [320, 160],[0, 320]])

    
    def get_data(self):
        """
        return the validation and training data sets
        """
        return self.training_samples, self.validation_samples
    
    def load_data(self, valid_size = 0.2, center = True, left= False, right = False):
        """
        this method returns the loaded data set
        """
        print("read csv file")
        self.read_csv_file()
        
        cleaned_data=[]
        low_angle_data = [] 
        print("clean data")
        # clean Data of too many low angle lines    
        for line in self.csv_file:
            angle = float(line[3])
            if abs(angle) >= 0.10:
                cleaned_data.append(line)
            else:
                low_angle_data.append(line)
                
        # add some low angles back to the test set
        x, splitted = train_test_split(self.csv_file, test_size=0.3)        
        for line in splitted:
                cleaned_data.append(line)   
        
        
        self.csv_file = cleaned_data
        
        self.training_samples, self.validation_samples = train_test_split(self.csv_file, test_size=valid_size)
        print(self.training_samples[0])

        #set the status of which camera to use
        if center == False:
            print("removing center cam")
            self.active_cams["center"] = center
        if left == False:
            print("removing left cam")
            self.active_cams["left"] = left
        if right == False:
            print("removing right cam")
            self.active_cams["right"] = right
        
    def get_csv_file(self):
        return self.csv_file
    
                
    def read_csv_file(self):
        """
        this method reads the csv files in the given path directory
        """
        with open(self.path+"//driving_log.csv") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.csv_file.append(line)
        print("driving_log.csv successfully loaded")
        
    def transform_img(self,img):
        """
        transform the image towars the wished pattern
        """
        print(type(img))
        print(img)
        img_size = (img.shape[1],img.shape[0])
        
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        
        #plt.imshow(warped)
        #plt.show()
        return warped
    
    def batch_loader(self, target = "train", batch_size = 64):
        """
        this method will load only a batch of images
        target: define if validation or training data should be loaded
        batch_size: defines the batchsize of the given data_set
        """
        if target == "train":
            samples = self.training_samples
            num_samples = len(samples)
            while 1: # Loop forever so the generator never terminates
                sklearn.utils.shuffle(samples)
                for offset in range(0, num_samples, batch_size):
                    batch_samples = samples[offset:offset+batch_size]
                    images = []
                    angles = []
                    for batch_sample in batch_samples:
                        if self.active_cams["center"]:
                            name = batch_sample[0]
                            center_image = (mpimg.imread(name))
                            center_angle = float(batch_sample[3])                   
                            images.append(center_image)
                            angles.append(center_angle)
                        if self.active_cams["right"]:
                            name = batch_sample[2]
                            right_image = (mpimg.imread(name))
                            right_angle = (float(batch_sample[3])-0.2)
                            images.append(right_image)
                            angles.append(right_angle)
                        if self.active_cams["left"]:
                            name = batch_sample[1]
                            left_image =(mpimg.imread(name))
                            left_angle = (float(batch_sample[3])+0.2)
                            images.append(left_image)
                            angles.append(left_angle)
                            
                    for i in range(0,len(images)):
                        r = np.random.random()
                        if  r <= 0.6:
                            images.append(cv2.flip(np.copy(images[i]),+1))
                            angles.append(angles[i]*-1)
                        elif r > 0.6:
                            #todo random brithness
                            images.append(self.brightness_image(images[i],np.random.random()))
                            angles.append(angles[i])   
                        else:
                            images.append(self.brightness_image(cv2.flip(np.copy(images[i]),+1),np.random.random()))
                            angles.append(angles[i]*-1)
                    # trim image to only see section with road
                    X_train = np.array(images)
                    y_train = np.array(angles)
                    
                    
                    yield sklearn.utils.shuffle(X_train, y_train)
                    
        elif target == "valid":
            samples = self.validation_samples
            
            num_samples = len(samples)
            while 1: # Loop forever so the generator never terminates
                sklearn.utils.shuffle(samples)
                for offset in range(0, num_samples, batch_size):
                    batch_samples = samples[offset:offset+batch_size]
                    images = []
                    angles = []
                    for batch_sample in batch_samples:
                        if self.active_cams["center"]:
                            name = batch_sample[0]
                            center_image = (mpimg.imread(name))
                            center_angle = float(batch_sample[3])                  
                            images.append(center_image)
                            angles.append(center_angle)
                            
                    # trim image to only see section with road
                    X_train = np.array(images)
                    y_train = np.array(angles)
                    
                    
                    yield sklearn.utils.shuffle(X_train, y_train) 
                       
    def brightness_image(self, img, rand=0.5):
        amount = (rand - 0.5) * 191  # * 255 * 0.75
        img = img + amount
        img = np.clip(img, 0, 255)
        return img

