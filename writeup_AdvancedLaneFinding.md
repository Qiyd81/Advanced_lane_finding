## Writeup

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
[image0]: ./test_images/original_image_with_src.jpg "straight_lines1 with source points drawed"
[image1]: ./output_images/image_undistort "Undistorted"
[image2]: ./output_images/image_threshold.jpg "Color_gradient thresholded"
[image3]: ./output_images/image_region.jpg "Region masked"
[image4]: ./output_images/image_warp.jpg "Perspective Warped"
[image5]: ./output_images/image_with_search_area "Around_poly drawed"
[image6]: ./output_images/final_output "Unwarped Image"
[video1]: ./project_solution.mp4 "Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients.
Provide an example of a distortion corrected calibration image, it is referenced as [image1] above.

The code for this step is contained in lines 55 through 100 of the file called `Advanced_lane_finding_myown.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image [image0] using the `cv2.undistort()` function and obtained this result:
[image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
use the function 'undistort_img' with the coefficience dict 'dist_pickle' calculated before to output the undistorted image [image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds and combined thresholds together to generate a binary image ('color_gradient' thresholding steps at lines 112 through 134; and 'combined_thresholds' steps at lines 203 through 267 in `Advanced_lane_finding_myown.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

[image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_birdview_transform(img,M)`, which steps at lines 162 through 182 in the file `Advanced_lane_finding_myown.py`.
The output image is [image3].  The `perspective_birdview_transform(img,M)` function takes as inputs an image (`img`), and also the matrix 'M' which comes from the function perspective_matrix(src_pts, dst_pts). aAnd the source (`src_pts`) and destination (`dst_pts`) points are chosen manually.
I didn't chose the hardcode the source and destination points in the following manner, but will try next time:
### below is the reference I will try next time:
```python
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
```
'''
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |
###

The src_pts and dst_pts I manually chose as follows:
| Source        | Destination   |
|:-------------:|:-------------:|
| [230,720]     | [230,720]     |
| [560,470]     | [230,0]       |
| [720,470]     | [1100,0]      |
| 1100,720]     | [1100,720]    |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

[image0]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The code steps at lines 273 through 352, to identify the lane_line pixels, and steps at lines 355 through 383, to fit their polynomials .

Then I fit my lane lines with a 2nd order polynomial and search a margin area 'search_around_poly()' kinda like this:

[image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 474 through 497 in my code in `Advance_lane_finding_myown.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in line 534 or line 572(test on single image) in my code in `Advanced_lane_finding_myown.py` .  Here is an example of my result on a test image:

[image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
