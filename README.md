# **Behavioral Cloning for Self-driving Car** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Dataset Summary
The data I used for training the model are collected from track 1 by mannually driving for one clockwise loop and counter-clockwise loop. Additionally, I also include dataset from Udacity.

| Data Type            |     Value	     	 | 
|:--------------------:|:-----------------:| 
| Training Data Size   | 17950   					 | 
| Validation Data Size | 4488	             |
| Test Data Size       | 2494	             |
| Image Shape          | 160 x 320 x 3		 |

The streering angles distribution are shown in Figure 1. As we can see, for the track 1, there are more negative steering angles while counter-clockwise loop has more positive steering angles. Overall, the data is not evenly distributed and the majority of steering angles are around 0.
<p align="center">
  <img src="report/angles_distribution.jpg" width="1000" height="300"/>
  <br>
  <em>Figure 1: Distribution of Steering Angles</em>
</p>

The data is evenly distributed for left, center and right images. Sample images are shown as below:
<p align="center">
  <img src="report/sample_camera_imgs.jpg" width="1000" height="300"/>
  <br>
  <em>Figure 2: Sample Camera Images</em>
</p>

## Data Preprocessing

Looking at the sample images, the upper and lower portions are irrelevant. So we are cropping those out by adding a cropping layer to the model:
```python
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
```
The images after cropping are shown as below:
<p align="center">
  <img src="report/cropped_imgs.jpg" width="1000" height="300"/>
  <br>
  <em>Figure 3: Cropped Sample Camera Images</em>
</p>

Another data augument I did is to flip images and angles. In this way, we can increase the data size to 2x.

## Model Architecture

The architecture of the network is shown as below:

| Layer         	    	|     Description	        		            			| 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 160 x 320 x 3   						                	| 
| Cropping             	| Upper 50 and lower 20 pixels are cropped out 	| 
| Convolution 5x5     	| 24 filters, 2x2 stride, valid padding         |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Dropout               | 0.2 dropout rate                              |
| Convolution 5x5 	    | 36 filters, 2x2 stride, valid padding       	|
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Dropout               | 0.2 dropout rate                              |
| Convolution 3x3 	    | 48 filters, 2x2 stride, valid padding       	|
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Dropout               | 0.2 dropout rate                              | 
| Flatten 	        	  |             									                |
| Fully connected		    | output 1024, ReLu Activation					        |
| Fully connected		    | output 256, ReLu Activation			  		        |        								            
| Fully connected		    | output 64, ReLu Activation                    |
| Fully connected		    | output 1        								              |
| MSE                   |                                               |

## Training Strategy

To train this network, we use ```Adam``` optimizer with learning rate of **0.001** for **20** epochs. To get the best model, we save the checking point after each epoch with best validation error. This can be easily implemented using keras:
```python
 checkpointer = ModelCheckpoint(filepath='ckpt/model.hdf5', verbose=1, save_best_only=True)
 adam = Adam(lr=0.001)
 
 model.compile(loss='mean_squared_error', optimizer=adam)
 hist  = model.fit_generator(
      generator = train_generator,
      steps_per_epoch = math.ceil(len(train_samples) / 32.0),
      validation_data = validation_generator,
      validation_steps = math.ceil(len(validation_samples) / 32.0),
      epochs = 20,
      callbacks=[checkpointer],
   )
```
Besides, since we have more than 20k images, it is easy to overwhelm memory usage. To avoid that, we use python generator to produce image data for each batch.

## Evaluation Model
<p align="center">
  <img src="report/training_val_loss.jpg" width="500" height="300"/>
  <br>
  <em>Figure 4: Training and validation loss</em>
</p>

The training loss and validation loss during training process are shown above. The final test loss is **0.017658**.

## Improvements

At first the trained car had problem to take large turns, especially the sharp turns after bridge. The is due to training data bais. As Figure 1 shows, most of angles are around 0, therefore the car always tended to make smaller turns. 

To avoid this, we use sub-sampling in our generator to get larger angles with higher probabilities. Besides, by utilizing left and right cameras, we will have more data with higher steering angles. For left images, we will give ```angle = center_angle -0.25```. Similarly, we will assign ```angle = center_angle + 0.25``` for right images. The generator code is shown as below:

```python
   HIGHER_ANGLE_THRESHOLD = 0.25
   HIGHER_ANGLE_PROBABILITY = 0.5
   """
   Return (image_path, angle) with sub-sampling.
   """
   def get_one_sample(sample):
      should_get_higher_angle = random.uniform(0, 1) <= HIGHER_ANGLE_PROBABILITY
      for i in range(0, 3):
          steering_bias = 0.0
          if i == 1:
              steering_bias = 0.25
          elif i == 2:
              steering_bias = -0.25
 
          image_path = sample[i]
          angle = steering_bias + float(sample[3])
          if should_get_higher_angle and abs(angle) < HIGHER_ANGLE_THRESHOLD:
              continue
          return (image_path, angle)
```

With this sub-sampling technique, the car can make lager turns easily.
