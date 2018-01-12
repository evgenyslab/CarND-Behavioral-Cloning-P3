# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Model Architecture and Training Strategy

### Architecture
The model architecture was based on the highly acclaimed [NVidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The input layer to the model consisted of a normalized image that was cropped with a Keras cropping function to remove the sky and car hood.

| Layer | Description|
|-------|-----------:|
| Input Image | Normalize|
| Crop | Row cropping|
| Convolution | Depth 24, relu|
| Convolution | Depth 36, relu|
| Convolution | Depth 48, relu|
| Convolution | Depth 64, relu|
| Convolution | Depth 64, relu|
| Flatten| |
| Fully Connected | 100 |
| Dropout | 70% | 
| Fully Connected | 50 |
| Dropout | 85% |
| Fully Connected | 10 |
| output | 1 |


Two dropout layers were introduced to the NVidia model to improve model robustness and reduce overfitting.

The model used an adam optimized. The only tuned paramter was the number of ephocs which was set to 5. The training data was split into 80-20 ratio for training/validation.

A generator was used to load images from the storage drive to avoid loading all images into memory directly.


### Strategy
The provided training data in conjunction with the above-described model was capable of smoothly driving the vehicle around the test track. The model could not drive in the second track. The model's response was very smooth which suggested that the model was overfitting the provided training data.

To attempt a more robust approach, new training data was gathered using a controller input instead of mouse and keyboard. A total of 6 loops were gathered, 3 in the forward direction, and 3 in the backward direction. 1 loop in each direction was gathered in 'recovery' mode were the vehicle was driven with swaying pattern to simulate recovery towards the center. The new training data consisted of 7478 images from all 6 loops (in center channel).

The model was trained on an NVidia GTX1070 video card with 5 epochs. The resulting model with the new data produced less smooth motion than the provided training data. Specifically, the new model gently swayed the vehicle from side to side. This suggests that the newly trained model attempted to position the vehicle such that the road edges became more clear in the view.

The new model was capable of continuously remaining on the track indefinately. The model did not perform on the suplementally track.