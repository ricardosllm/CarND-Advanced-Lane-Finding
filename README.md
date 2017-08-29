# CarND-Advanced-Lane-Finding

The goal of this project is to identify the road lanes in a video stream of a forward facing camera mounted centeraly in a moving vehicle.

We'll be using image manipulation techniques to extract enough information from each fram, or image, from the video and identify the lane lines, the radius of curvature and the distance from the camera to the center line of the road.

![alt text](videos/project_video_augmented.gif "Result")

## Project Structure

- `camera_cal/` Directory with calibration images
- `test_images/` Directory with test images 
- `output_images/` Directory with test images with augmented overlay
- `videos/` Directory with input and output videos 
- `Advanced-Lane-Finding.ipnyb` Jupyter notebook with all the project code and example images
- `README.md` Projecte writeup (you're reading it)

## Project Overview

In order to detect the lane lines in a video stream we must accomplish the folowing:

- **Camera Calibration** Calibrate the camera to correct for image distortions. For this we use a set of chessboard images, knowing the distance and angles between common features like corners, we can calculate the tranformation functions and apply them to the video frames.

- **Color Transform** We use a set of image manipulation techniques to accentuate certain features like lane lines. We use color space transformations, like from RGB to HLS, channel separation, like separating the S channel from the HLS image and image gradient to allow us to identify the desired lines.

- **Perspective Transform** 



## Camera Calibration

Before we can use the images from the front facing camera we must calibrate it so we can correctly measure distances between features in the image.

To do this we first must find the calibration matrix and distortion coefficients for the camera given a set of chessboard images.
