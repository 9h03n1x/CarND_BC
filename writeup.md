# **Behavioral Cloning** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./center_2018_12_28_22_31_29_798.jpg "center img"
[image3]: ./right_2018_12_30_19_05_26_286.jpg "right img"
[image4]: ./left_2019_01_07_22_45_39_156.jpg "left img"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* main_project.py containing the script to create and train the model
* data_loader.py containing the generator and methods to load the image and angle data also the augmenting methods
* drive.py for driving the car in autonomous mode with adaption to my local system
* model1.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model1.h5
```

#### 3. Submission code is usable and readable

The main_project.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

the choosen architecture is the nvidia model with dropout:

0.  Preprocessing the given data norm and crop                        
1st Layer Conv2D 1x1 with max strides 1x1                           
2nd Layer Conv2D 5x5 with max strides 2x2                           
3rd Layer Conv2D 5x5 with max strides 2x2                           
4th Layer Conv2D 5x5 with max strides 2x2                           
5th Layer Conv2D 3x3 with max strides 1x1                           
6th Layer Conv2D 3x3 with max strides 1x1                           
7th Layer Fully Connected with relu activation dropout              
6th Layer Fully Connected with relu activation dropout              
7th Layer Fully Connected with relu activation dropout              
8th Layer Fully Connected with relu activation dropout                 
9th Layer Output layer for steering angle prediction                

before creatin the modell the data loader is created

	 #init the data loader
    dl = data_loader(path)
    # load the data and activate the different cameras
    dl.load_data(valid_size = 0.2,center = True, left= True, right = True )

get the data from the data loader and create the generators for the training data and the validation data

    train_samples, validation_samples = dl.get_data()
    batch_size = 16
    training_gen = dl.batch_loader(target = "train", batch_size= batch_size)
    valid_gen = dl.batch_loader(target = "valid", batch_size= batch_size)


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and 3x3 filters (main_project.py lines 68-115).
The model architecture is the the same as the nvidia model with dropouts

    model = Sequential()
    #0th Layer Data Preporcessing (cropping and normalizing ,input_shape=(320,160,3)
    model.add(Cropping2D(cropping=((60,15), (0,0)), input_shape=(160,320,3)))
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


The model includes RELU layers to introduce nonlinearity and dropouts to prevent the model from overfitting, and the data is normalized in the model using a Keras lambda layer (code line 72). 


#### 3. Model parameter tuning

The model used an adam optimizer, with an inital learnreat of lr=1e-4 (model.py line 121).
I chose a correction factor of 0.3 for the left and right camera
a batchsize of 8
and a dropout of 0.3

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I drove about 4 rounds arround the trck and used all thre cameras, I added/subtracted a correction factor to the angles related to right/left camera. I also augmented the Training Data, some with random brightness, some flipped and some both


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to at first to play arround with the convolutional layers and the fully connected layer, in the end I had some simething similar to the nvidia architecture, so I adapted to the established architecture

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set at a rate of 80/20. during training I always had a very low accuracy, strangely the lower the accuracy the better the model
To prevent the model from overfitting I added dropouts to the fully connected layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially in corners with rare situation. To improve the driving behavior in these cases, I recorded onyl the critical scenes again, to have more data to learn that situation.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one trying center lane driving. Here is an example image of center lane driving, it did not always work very good, I used all three cameras for the training, but added a correction factor to the angles of the left and right camera:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

Before using the data I also cleaned the data of too many low angle images:
 	
	# clean Data of too many low angle lines    
	for line in self.csv_file:
	    angle = float(line[3])
	    if abs(angle) >= 0.10:
		cleaned_data.append(line)
	    else:
		low_angle_data.append(line)

	# add some low angles back to the test set
	x, splitted = train_test_split(self.csv_file, test_size=0.2)        
	for line in splitted:
		cleaned_data.append(line)   


	self.csv_file = cleaned_data
        

To augment the data set, I also flipped images and angles thinking that this would save me time driving the track the other way arround, also i had some images with a random brightness

	for i in range(0,len(images)):
		r = np.random.random()
		if  r <= 0.6:
		    images.append(cv2.flip(np.copy(images[i]),+1))
		    angles.append(angles[i]*-1)
		elif r > 0.6:
		    #todo random brithness
		    images.append(self.brightness_image(images[i],np.random.random()))
		    angles.append(angles[i])
		    
With this i could double my data set with new and different data. This was all done in the generators and only to the training data not to the validation data, here I only used the center camera images. before returning the Batch, the images in the batch where shuffeled again

![alt text][image6]
![alt text][image7]



After the collection process, I had 30000 number of data points.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by ... 
I used an adam optimizer so that manually training the learning rate wasn't necessary. Even though I tried training with a set learn rate
