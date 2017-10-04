## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undist_chessboard.jpg "Original"
[image2]: ./undist_chessboard.jpg "Undistorted"
[image3]: ./Undistorted_Warped_Image.jpg "Warped chess board image"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./Undistorted_Warped_Image_road.jpg "Warp Example"
[image5]: ./lane-line_pixels.jpg "Fit Visual"
[image6]: ./result.jpg "Output"
[video1]: ./project_out.mp4 "Video"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Pipeline (single images)

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in function fist section of IPython notebook located in "./advanced_lane_finding.ipynb". I first read in the calibration images. Using `cv2.findChessboardCorners()` function from opencv library I find the chessboard corners (for an 9x6 board). Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

Then I calibrate camera using object points, image points, and the shape of the grayscale image with the function `cv2.calibrateCamera()`. Finally, using the camera calibration matrix obtained from `cv2.calibrateCamera()` function I undistort a given test image using `cv2.undistort()`.

Following is the result obtained after applying distortion correction to a test image.


![alt text][image1]
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps is done in the function `color_threshold` in the 3rd code cell of the IPython notebook. I first convert a given image to HLS color space using `cv2.cvtColor()`. Next, I take a derivative in x direction of luminance channel of HLS color space using a Sobel filter. I then find the absolute x derivative to accentuate lines away from horizontal. Then I threshold x gradient. I also threshold color channel (i.e. S channel). Then I stack the results of clor threshold and gradient threshold to obtain binary image.

Following images depicts the results of the binary image after using `color_threshold` function.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in function called `corners_unwarp()`,  in the 2nd code cell of the IPython notebook in the file `advanced_lane_finding.ipynb`.  The `corners_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. 
I chose the hardcode the source and destination points in the following manner:

``python
src = np.float32(
[[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
[((img_size[0] / 6) - 10), img_size[1]],
[(img_size[0] * 5 / 6) + 60, img_size[1]],
[(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
[[(img_size[0] / 4), 0],
[(img_size[0] / 4), img_size[1]],
[(img_size[0] * 3 / 4), img_size[1]],
[(img_size[0] * 3 / 4), 0]])
``

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Following image shows the bird eye view of lane line obtained by `corners_unwarp()` function.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


The code for this part is in the function `my_pipeline()` included in the 5th code cell of the IPython notebook in the file `advanced_lane_finding.ipynb` . To identify lane-line pixels and fit their positions with a polynomial first I take a histogram of the bottom half of the binary warped image. The I create an output image to draw on and visualize the result. Then we find the peak of the left and right halves of the histogram, these will be the starting point for the left and right lines.

Next using widow-based processing I segment the image into the multiple windows and process each window starting form bottom of the image to the top to identify the lane line in the entire binary warped image. I put the indices identified as left lane line and right lane line in two separte arrays of `left_lane_inds` and `right_lane_inds`. Then based on nonzero indices in the binary warped image we identify the x and y pixel value of these left and right lane lines. 

Then using Numpy function `np.polyfit()` I fit a second order polynomial to x and y pixels of left and right lane lines.

Follwing image shows the results obtained from using this method to identify lane line in a given binary warped image.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `advanced_lane_finding.ipynb` in the function `my_pipeline()`. 

Now that we have a thresholded image, where we have estimated which pixels belong to the left and right lane lines (shown in red and blue, respectively), and I've fit a polynomial to those pixel positions. Next I compute the radius of curvature of the fit.  I've calculated the radius of curvature based on pixel values, so the radius we are reporting is in pixel space, which is not the same as real world space. So I actually need to repeat this calculation after converting our x and y values to real world space. This involves measuring how long and wide the section of lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane in the field of view of the camera. For this project, the lane is about 30 meters long and 3.7 meters wide.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `my_pipeline()`. 

Once I have a good measurement of the line positions in warped space, I project the result back down onto the road. Let's suppose, we have a warped binary image called `binary_warped`, and you have fit the lines with a polynomial and have arrays called `ploty`, `left_fitx` and `right_fitx`, which represent the x and y pixel values of the lines. You can then project those lines onto the original image as follows:

``
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
``

`Minv` represent the inverse of perspective transform function. The result of of this process in depicted in the following image. 

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problem occur in the areas where lane line aren't as vivid to get detected by my color threshold function. like for example in the shaded areas. The color thresholding function should get improved for this matter.

