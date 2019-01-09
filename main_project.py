'''
Created on 29.12.2018

@author: Nicco
'''

import tensorflow as tf
from keras import backend as k


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D
#from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Cropping2D
#from keras.preprocessing.image import random_brightness
from keras.optimizers import Adam
#import numpy as np
#import pickle

from tensorflow.python.client import device_lib
k.tensorflow_backend._get_available_gpus()
print(device_lib.list_local_devices())


from data_loader import data_loader

if __name__ == '__main__':
    # this is needed due to hardware setup
    config = tf.ConfigProto(log_device_placement=False)
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of 70% the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))
    
    path = "C://MyWorkspace//workspace//BC_CarND//training_data"
    
    dl = data_loader(path)
    #x_train, y_train, x_valid, y_valid = dl.load_data(valid_size = 0.2)
    dl.load_data(valid_size = 0.2,center = True, left= True, right = True )
    
    train_samples, validation_samples = dl.get_data()
    batch_size = 16
    training_gen = dl.batch_loader(target = "train", batch_size= batch_size)
    valid_gen = dl.batch_loader(target = "valid", batch_size= batch_size)
    
    
    #######################################################################
    ################ Keras Model  #########################################
    # Preprocessing the given data norm and crop                          #
    # 1st Layer Conv2D 1x1 with max strides 1x1                           #
    # 2nd Layer Conv2D 5x5 with max strides 2x2                           #
    # 3rd Layer Conv2D 5x5 with max strides 2x2                           #
    # 4th Layer Conv2D 5x5 with max strides 2x2                           #
    # 5th Layer Conv2D 3x3 with max strides 1x1                           #
    # 6th Layer Conv2D 3x3 with max strides 1x1                           #
    # 7th Layer Fully Connected with relu activation dropout              #
    # 6th Layer Fully Connected with relu activation dropout              #
    # 7th Layer Fully Connected with relu activation dropout              #
    # 8th Layer Fully Connected with relu activation dropout              #   
    # 9th Layer Output layer for steering angle prediction                #
    ################ Keras Model  #########################################
    #######################################################################
    
    model = Sequential()
    #0th Layer Data Preporcessing (cropping and normalizing ,input_shape=(320,160,3)
    model.add(Cropping2D(cropping=((60,15), (0,0)), input_shape=(160,320,3)))
    #model.add(Lambda(tf.image.rgb_to_grayscale))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    
    model.add(Conv2D(3,kernel_size=(1, 1), strides=(1, 1)))
    model.add(Activation('relu'))
    #1st Layer Conv2D with max Pooling 2x2
    model.add(Conv2D(24,kernel_size=(5, 5) ,strides=(2,2)))
    model.add(Activation('relu'))

    #2nd Layer Conv2D with max Pooling 2x2
    model.add(Conv2D(36, kernel_size=(5, 5),strides=(2,2)))
    model.add(Activation('relu'))

    #3rd Layer Conv2D with max Pooling 2x2
    model.add(Conv2D(48, kernel_size=(5, 5),strides=(2,2)))
    model.add(Activation('relu'))

    #4th Layer Conv2D with max Pooling 2x2
    model.add(Conv2D(64, kernel_size=(3, 3),strides=(1,1)))
    model.add(Activation('relu'))

    #5th Layer Conv2D with max Pooling 2x2
    model.add(Conv2D(64, kernel_size=(3, 3),strides=(1,1)))
    model.add(Activation('relu'))
        
    #5th Layer Fully Connected with elu activation
    model.add(Flatten())
    #model.add(Dropout(0.25))
    model.add(Activation('relu'))

    #6th Layer Fully Connected with elu activation
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    #7th Layer Fully Connected with  elu activation
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    #8th Layer Fully Connected with  elu activation
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    #9th Layer Fully Connected 
    model.add(Dense(1))
    #model.add(Activation('relu'))
    
    #model.compile('adam', 'mse', ['accuracy'])
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #adadelta=optimizers.Adadelta()
    optimizer = Adam(lr=1e-4)
    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mae','acc'])
    model.fit_generator(training_gen, samples_per_epoch=len(train_samples)*2, validation_data=valid_gen, 
            nb_val_samples=len(validation_samples), nb_epoch=5)
    
    model.save("C://MyWorkspace//workspace//BC_CarND//model//model1.h5", overwrite =True)
    #######################################################################
    ################ Keras Model  #########################################
    #######################################################################