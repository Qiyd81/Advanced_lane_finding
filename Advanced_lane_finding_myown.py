# Project Steps
#
# Steps we’ve covered so far:
#
# Camera calibration
# Distortion correction
# Color/gradient threshold
# Perspective transform
# After doing these steps, you’ll be given two additional steps for the project:
#
# Detect lane lines
# Determine the lane curvature

# import modules, skimage.morphology used for isolating noise lines.
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from skimage import morphology
from collections import deque

# Define read the pictures
def read(img):
    return cv2.imread(img)

# Define show one picture
def show(img,title,color=1):
    '''
    :param img: input image
    :param title: image title
    :param color: 1 for color image, 0 for gray image
    '''
    if color:
        plt.imshow(img)
    else:
        plt.imshow(img,cmap='gray')
    plt.title(title)
    plt.axis('on')
    plt.show()
# Define show 2 pictures
def show_2pics(image1, title1, image2, title2):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image1)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(image2)
    ax2.set_title(title2, fontsize=30)
    plt.show()

# calibrate the camera, modified from https://github.com/Qiyd81/CarND-Camera-Calibration.git/camera_calibration.ipynb
def calibrate_camera(folder, nx, ny):
    '''
    :folder: input image folder
    :nx: number of x corners of chessboard
    :ny: number of y corners of chessboard
    :return: a dictionary contains the coefficients for undistortion usage
        ret: RMS varients
        mtx: camera matrix
        dist: distortion coefficience
        rvecs: rotating vectors
        tvecs：transfer vectors
    '''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Step through the folder and search for chessboard corners
    for fname in folder:
        img = read(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (gray.shape[1], gray.shape[0])

    # Do camera calibration given object points and image points, ret: calibrated
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # Save the camera calibration result for later use (we don't care about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    return dist_pickle

def undistort_img(img, dist_pickle):
    undist = cv2.undistort(img, dist_pickle['mtx'], dist_pickle['dist'], None, dist_pickle['mtx'])
    return undist

# Use the camera_cal pictures to calibrate the dist_pickle coefficiences.
nx = 9
ny = 6
camera_cal = glob.glob("camera_cal/calibration*.jpg")
dist_pickle = calibrate_camera(camera_cal, nx, ny)
# #test undistort_img function
# image_chessboard = cv2.imread('./camera_cal/calibration1.jpg')
# image_chessboard_undistort = undistort_img(image_chessboard, dist_pickle)
# show(image_chessboard_undistort, 'image_chessboard_undistort')

# TODO: to write the output image to output_images folder, imwrite not working yet
#cv2.imwrite('./output_images/test_image_undistort', test_image_undistort)

# color_gradient function to convert color image to HlS format, and then find gradients.
def color_gradient(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # # Stack each channel
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = np.zeros_like(s_binary)
    color_binary[(sxbinary == 1) | (s_binary == 1)] = 1
    return color_binary

# mask for interesting area lane lines
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def perspective_matrix(src_pts=np.float32([[230,720],[560,470],[720,470],[1100,720]]),dst_pts=np.float32([[230,720],[230,0],[1100,0],[1100,720]])):
    '''
    perspective_matrix: the coefficience for convert between source_points and destination_points
    args:source and destiantion points
    return M and Minv
    '''
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts,src_pts)
    return {'M':M,'Minv':Minv}

def perspective_birdview_transform(img,M):
    '''
    Transform image to birdeye view
    img:binary image
    M:transformation matrix
    return a wraped image
    '''
    img_sz = (img.shape[1], img.shape[0])
    img_warped = cv2.warpPerspective(img, M, img_sz, flags=cv2.INTER_LINEAR)

    return img_warped

#test perspective_matric function
# src_pts = np.float32([[230,720],[560,470],[720,470],[1100,720]])
# dst_pts = np.float32([[230,720],[230,0],[1100,0],[1100,720]])
# transform_matrix = perspective_matrix(src_pts,dst_pts)

# # # #test the source points position
# test_straightline = cv2.imread('./test_images/straight_lines1.jpg')
# plt.imshow(test_straightline)
# plt.plot(230,720,'.')
# plt.plot(560,470,'.')
# plt.plot(720,470,'.')
# plt.plot(1100,720,'.')
# plt.show()

# Combined threshold
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# combine all 3 different thresholds
def combined_thresholds(image, ksize=9):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 120))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.5, np.pi/2))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

# #wrong step//test image after combined threshold and warp function.
# warp_img_combined_threshold = perspective_birdview_transform(img_combined_threshold,transform_matrix['M'])
# show_2pics(img_combined_threshold, 'img_combined_threshold', warp_img_combined_threshold, 'warp_img_combined_threshold')

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    vehicle_pos = midpoint - int(round((leftx_base + rightx_base)//2))
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img, vehicle_pos, leftx_base, rightx_base


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img, vehicle_pos,leftx_base, rightx_base = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, vehicle_pos

# #test processs_image function
# input_image = cv2.imread('./test_images/test1.jpg')
# output_image = process_image(input_image)[0]
# show(output_image,'output_image',color=0)

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #calculate the left. right fit coefficience
    left_fit = fit_polynomial(binary_warped)[4]
    right_fit = fit_polynomial(binary_warped)[5]
    vecicle_pos = fit_polynomial(binary_warped)[6]
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return result, left_fitx, right_fitx, ploty, vecicle_pos
#test search_around_poly function
# around_poly = search_around_poly(image_warp)
# plt.imshow(around_poly)
# plt.show()

def measure_curvature_real(ploty, left_fitx, right_fitx, vehicle_pos):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 12 / 450  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 910  # meters per pixel in x dimension

    # Make sure to feed in your real data instead in your project!
    ploty_real = ploty*ym_per_pix
    left_fit_real = np.polyfit(ploty*ym_per_pix,left_fitx*xm_per_pix,2)
    right_fit_real = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix,2)
    vehicle_pos_real = vehicle_pos * xm_per_pix
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty_real)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_real[0] * y_eval * ym_per_pix + left_fit_real[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_real[0])
    right_curverad = ((1 + (2 * right_fit_real[0] * y_eval * ym_per_pix + right_fit_real[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_real[0])

    return left_curverad, right_curverad, vehicle_pos_real

# #test measure curvature function
# left_fitx = process_image('./test_images/test1.jpg')[1]
# right_fitx = process_image('./test_images/test1.jpg')[2]
# print(left_fitx)
# print(right_fitx)
# curvature = measure_curvature_real(image_warp, left_fitx, right_fitx)
# print(curvature)

#TODO following class
# Define a class to receive the characteristics of each line detection
class Line():
    '''class to receive the charateristics of each line detection'''
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        # was the line detected in the last iteration?
        self.detected = False
        # the base position of left and right line
        self.left_base = None
        self.right_base = None
        # x values of the last n fits of the line
        self.recent_xfitted = deque([], maxlen=self.buffer_size)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = {'a0': deque([], maxlen=self.buffer_size),
                        'a1': deque([], maxlen=self.buffer_size),
                        'a2': deque([], maxlen=self.buffer_size)}
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = deque([], maxlen=self.buffer_size)
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = deque([], maxlen=self.buffer_size)
        #y values for detected line pixels
        self.ally = deque([], maxlen=self.buffer_size)

def sanity_check(left_fitx, right_fitx, leftx_base, rightx_base, left_curverad, right_curverad):
    '''
    check if current detected line match with previous lines, and also are reasonable line
    '''
    flag_line = True
    # fit_plonomial() return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, vehicle_pos, leftx_base, rightx_base
    # first check the left and right line distance
    base_dist = abs(leftx_base - rightx_base)
    line_dist = abs(np.mean(left_fitx) - np.mean(right_fitx))
    if abs(base_dist-line_dist) > 50 or base_dist<600 or line_dist<600 or abs(left_curverad - right_curverad)>100: # add
        flag_line = False
    return flag_line

line_left = Line(buffer_size=10)
line_right = Line(buffer_size=10)
def pipe_line(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)
    # Use the dist_pickle coefficiences to undistort the image.
    global dist_pickle
    image_undistort = undistort_img(img, dist_pickle)
    #show(image_undistort,'image_undistort')
    # Use color_gradient to threshold the image to binary image
    image_threshold_color = color_gradient(image_undistort)
    image_threshold_combined = combined_thresholds(image_undistort)
    image_threshold = np.zeros_like(image_threshold_color)
    image_threshold[(image_threshold_color==1)|(image_threshold_combined==1)] = 1
    #show(image_threshold, 'image_threshold', color=0)
    #Create a mask
    ysize = image_undistort.shape[0]
    xsize = image_undistort.shape[1]
    vertices = [np.array([[np.int(xsize*0.2), np.int(ysize)], [np.int(xsize*0.4), np.int(ysize*0.6)], [np.int(xsize*0.6), np.int(ysize*0.6)], [xsize, np.int(ysize)]])]
    # Use the mask to sort out ROI
    image_region = region_of_interest(image_threshold, vertices)
    # image_region = morphology.remove_small_objects(image_region.astype('bool'), min_size=50, connectivity=2)
    #show(image_region, 'image_region', color=0)

    # Pespective transform the image
    image_warp = perspective_birdview_transform(image_region, perspective_matrix()['M'])

    #TODO append all the pixles parameter to the Line class

    leftx, lefty, rightx, righty, out_img, vehicle_pos, leftx_base, rightx_base = find_lane_pixels(image_warp)
    out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, vehicle_pos = fit_polynomial(image_warp)
    left_curverad, right_curverad, vehicle_pos_real = measure_curvature_real(ploty, left_fitx, right_fitx, vehicle_pos)

    if sanity_check(left_fitx, right_fitx, leftx_base, rightx_base, left_curverad, right_curverad) or len(line_left.recent_xfitted)==0:
        line_left.recent_xfitted.append(left_fitx)
        line_right.recent_xfitted.append(right_fitx)

    else:
        left_fitx = np.mean(line_left.recent_xfitted, axis=0)
        right_fitx = np.mean(line_right.recent_xfitted, axis=0)
        line_left.recent_xfitted.append(left_fitx)
        line_right.recent_xfitted.append(right_fitx)
        # left_fitx = line_left.recent_xfitted.pop()
        # line_left.recent_xfitted.append(left_fitx)
        # line_left.recent_xfitted.append(left_fitx)
        #
        # right_fitx = line_right.recent_xfitted.pop()
        # line_right.recent_xfitted.append(right_fitx)
        # line_right.recent_xfitted.append(right_fitx)
        # line_left.detected = True
        # line_right.detected = True



    # #if line_left.detected == True:
    # line_left.recent_xfitted.append(left_fitx)
    # # elif len(line_left.recent_xfitted) >= 1:
    # #     left_fitx = line_left.recent_xfitted.pop()
    # #     line_left.recent_xfitted.append(left_fitx)
    # #     line_left.recent_xfitted.append(left_fitx)
    #
    # # if line_right.detected == True:
    # line_right.recent_xfitted.append(right_fitx)
    # # elif len(line_right.recent_xfitted) >= 1:
    # #     right_fitx = line_right.recent_xfitted.pop()
    # #     line_right.recent_xfitted.append(right_fitx)
    # #     line_right.recent_xfitted.append(right_fitx)
    left_recent_xfitted_mean = np.mean(line_left.recent_xfitted, axis=0)
    right_recent_xfitted_mean = np.mean(line_right.recent_xfitted, axis=0)

    left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    line_pts = np.hstack((left_pts, right_pts))
    cv2.fillPoly(out_img, np.int_(line_pts), (0, 255, 0))
    #show(out_img, 'drawarea')
    #show(warp_output[0], 'output_image')
    curvature_real = measure_curvature_real(ploty, left_fitx, right_fitx, vehicle_pos)
    unwarp_output = cv2.warpPerspective(out_img, perspective_matrix()['Minv'], (img.shape[1], img.shape[0]))
    show(unwarp_output, 'drawback')
    final_output = cv2.addWeighted(img, 1, unwarp_output, 0.3, 0)
    #show(final_output, 'drawback')
    cv2.putText(final_output, 'Radius of curvature:' + str(round(curvature_real[0])) + 'm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), thickness=2)
    #TODO add the position of vechile
    cv2.putText(final_output, 'Position of vehicle:' + str(curvature_real[2]) + 'm', (100, 100),
                 cv2.FONT_HERSHEY_SIMPLEX,
                 1, (255, 255, 255), thickness=2)

    # print(curvature_real)
    # plt.plot(curvature_real[0], curvature_real[1])
    # plt.show()
    return final_output.astype('uint8')

# # #test the pipeline function with single image
# image = cv2.imread('./test_images/straight_lines1.jpg')
# pipe_line(image)






video_output = 'project_solution.mp4'
clipl = VideoFileClip("project_video.mp4")
white_clip = clipl.fl_image(pipe_line)
# %time white_clip.write_videofile(video_output,audio=False)
white_clip.write_videofile(video_output, audio=False)

#class Line functions
    # # copied function for filtering pixles not adjacent to main pixle streams.
    # def pixles_filter_distance(self, pixles_x, pixles_y):
    #     '''
    #     Mahalanobis Distance for bi-variate distribution
    #     D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x-\mu)}(latex)
    #     '''
    #     covariance_xy = np.cov(pixles_x, pixles_y, rowvar=0)
    #     inv_covariance_xy = np.linalg.inv(covariance_xy)
    #     pixle_mean = np.mean(pixles_x), np.mean(pixles_y)
    #     x_diff = np.array([x_i - pixle_mean[0] for x_i in pixles_x])
    #     y_diff = np.array([y_i - pixle_mean[1] for y_i in pixles_y])
    #     diff_xy = np.transpose([x_diff, y_diff])
    #
    #     md = []
    #     for i in range(len(diff_xy)):
    #         md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]), inv_covariance_xy), diff_xy[i])))
    #     return md
    #
    # def pixle_filters(self, pixle_x, pixle_y, distance_thresh=1.5):
    #     '''
    #     filter out pixels that the distance is further than distance_thresh
    #     '''
    #     pixles_distance = self.pixles_filter_distance(pixle_x, pixle_y)
    #     threshold = np.mean(pixles_distance) * distance_thresh
    #     pixle_keep_x, pixle_keep_y, pixle_remove = [], [], []
    #     for i in range(len(pixles_distance)):
    #         if pixles_distance[i] <= threshold:
    #             pixle_keep_x.append(pixle_x[i])
    #             pixle_keep_y.append(pixle_y[i])
    #         else:
    #             pixle_remove.append([pixle_x[i],pixle_y[i]])
    #     return pixle_keep_x, pixle_keep_y
    #
    # def find_lane_pixels(self, binary_warped):
    #     # Take a histogram of the bottom half of the image
    #     histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    #     # Create an output image to draw on and visualize the result
    #     out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    #     # Find the peak of the left and right halves of the histogram
    #     # These will be the starting point for the left and right lines
    #     midpoint = np.int(histogram.shape[0] // 2)
    #     leftx_base = np.argmax(histogram[:midpoint])
    #     rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #     vehicle_pos = midpoint - int(round((leftx_base + rightx_base) // 2))
    #     # HYPERPARAMETERS
    #     # Choose the number of sliding windows
    #     nwindows = 9
    #     # Set the width of the windows +/- margin
    #     margin = 100
    #     # Set minimum number of pixels found to recenter window
    #     minpix = 50
    #
    #     # Set height of windows - based on nwindows above and image shape
    #     window_height = np.int(binary_warped.shape[0] // nwindows)
    #     # Identify the x and y positions of all nonzero pixels in the image
    #     nonzero = binary_warped.nonzero()
    #     nonzeroy = np.array(nonzero[0])
    #     nonzerox = np.array(nonzero[1])
    #     # Current positions to be updated later for each window in nwindows
    #     leftx_current = leftx_base
    #     rightx_current = rightx_base
    #
    #     # Create empty lists to receive left and right lane pixel indices
    #     left_lane_inds = []
    #     right_lane_inds = []
    #
    #     # Step through the windows one by one
    #     for window in range(nwindows):
    #         # Identify window boundaries in x and y (and right and left)
    #         win_y_low = binary_warped.shape[0] - (window + 1) * window_height
    #         win_y_high = binary_warped.shape[0] - window * window_height
    #         win_xleft_low = leftx_current - margin
    #         win_xleft_high = leftx_current + margin
    #         win_xright_low = rightx_current - margin
    #         win_xright_high = rightx_current + margin
    #
    #         # Draw the windows on the visualization image
    #         cv2.rectangle(out_img, (win_xleft_low, win_y_low),
    #                       (win_xleft_high, win_y_high), (0, 255, 0), 2)
    #         cv2.rectangle(out_img, (win_xright_low, win_y_low),
    #                       (win_xright_high, win_y_high), (0, 255, 0), 2)
    #
    #         # Identify the nonzero pixels in x and y within the window #
    #         good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
    #                           (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    #         good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
    #                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    #
    #         # Append these indices to the lists
    #         left_lane_inds.append(good_left_inds)
    #         right_lane_inds.append(good_right_inds)
    #
    #         # If you found > minpix pixels, recenter next window on their mean position
    #         if len(good_left_inds) > minpix:
    #             leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    #         if len(good_right_inds) > minpix:
    #             rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    #
    #     # Concatenate the arrays of indices (previously was a list of lists of pixels)
    #     try:
    #         left_lane_inds = np.concatenate(left_lane_inds)
    #         right_lane_inds = np.concatenate(right_lane_inds)
    #     except ValueError:
    #         # Avoids an error if the above is not implemented fully
    #         pass
    #
    #     # Extract left and right line pixel positions
    #     leftx = nonzerox[left_lane_inds]
    #     lefty = nonzeroy[left_lane_inds]
    #     rightx = nonzerox[right_lane_inds]
    #     righty = nonzeroy[right_lane_inds]
    #     leftx, lefty = self.pixle_filters(leftx, lefty, distance_thresh=1.5)
    #     rightx, righty = self.pixle_filters(rightx, righty, distance_thresh=1.5)
    #
    #     return leftx, lefty, rightx, righty, out_img, vehicle_pos, leftx_base, rightx_base
    #
    # def fit_polynomial(self, binary_warped):
    #     # Find our lane pixels first
    #     leftx, lefty, rightx, righty, out_img, vehicle_pos, leftx_base, rightx_base = self.find_lane_pixels(binary_warped)
    #     # Fit a second order polynomial to each using `np.polyfit`
    #     left_fit = np.polyfit(lefty, leftx, 2)
    #     right_fit = np.polyfit(righty, rightx, 2)
    #
    #     # Generate x and y values for plotting
    #     ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    #     try:
    #         left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    #         right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #     except TypeError:
    #         # Avoids an error if `left` and `right_fit` are still none or incorrect
    #         print('The function failed to fit a line!')
    #         left_fitx = 1 * ploty ** 2 + 1 * ploty
    #         right_fitx = 1 * ploty ** 2 + 1 * ploty
    #
    #     ## Visualization ##
    #     # Colors in the left and right lane regions
    #     out_img[lefty, leftx] = [255, 0, 0]
    #     out_img[righty, rightx] = [0, 0, 255]
    #
    #     # Plots the left and right polynomials on the lane lines
    #     plt.plot(left_fitx, ploty, color='yellow')
    #     plt.plot(right_fitx, ploty, color='yellow')
    #
    #     return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, vehicle_pos, leftx_base, rightx_base
    #
    # def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
    #     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    #     left_fit = np.polyfit(lefty, leftx, 2)
    #     right_fit = np.polyfit(righty, rightx, 2)
    #     # Generate x and y values for plotting
    #     ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    #     ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    #     left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    #     right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #
    #     return left_fitx, right_fitx, ploty
    #
    # def search_around_poly(self, binary_warped):
    #     # HYPERPARAMETER
    #     # Choose the width of the margin around the previous polynomial to search
    #     # The quiz grader expects 100 here, but feel free to tune on your own!
    #     margin = 100
    #
    #     # Grab activated pixels
    #     nonzero = binary_warped.nonzero()
    #     nonzeroy = np.array(nonzero[0])
    #     nonzerox = np.array(nonzero[1])
    #
    #     # calculate the left. right fit coefficience
    #     left_fit = fit_polynomial(binary_warped)[4]
    #     right_fit = fit_polynomial(binary_warped)[5]
    #     vecicle_pos = fit_polynomial(binary_warped)[6]
    #     ### TO-DO: Set the area of search based on activated x-values ###
    #     ### within the +/- margin of our polynomial function ###
    #     ### Hint: consider the window areas for the similarly named variables ###
    #     ### in the previous quiz, but change the windows to our new search area ###
    #     left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
    #                                    left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
    #                                                                          left_fit[1] * nonzeroy + left_fit[
    #                                                                              2] + margin)))
    #     right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
    #                                     right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
    #                                                                            right_fit[1] * nonzeroy + right_fit[
    #                                                                                2] + margin)))
    #
    #     # Again, extract left and right line pixel positions
    #     leftx = nonzerox[left_lane_inds]
    #     lefty = nonzeroy[left_lane_inds]
    #     rightx = nonzerox[right_lane_inds]
    #     righty = nonzeroy[right_lane_inds]
    #
    #     # Fit new polynomials
    #     left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    #
    #     ## Visualization ##
    #     # Create an image to draw on and an image to show the selection window
    #     out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    #     window_img = np.zeros_like(out_img)
    #     # Color in left and right line pixels
    #     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #
    #     # Generate a polygon to illustrate the search window area
    #     # And recast the x and y points into usable format for cv2.fillPoly()
    #     left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    #     left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
    #                                                                     ploty])))])
    #     left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #     right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    #     right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
    #                                                                      ploty])))])
    #     right_line_pts = np.hstack((right_line_window1, right_line_window2))
    #
    #     # Draw the lane onto the warped blank image
    #     cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    #     cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    #     result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    #
    #     # Plot the polynomial lines onto the image
    #     plt.plot(left_fitx, ploty, color='yellow')
    #     plt.plot(right_fitx, ploty, color='yellow')
    #     ## End visualization steps ##
    #
    #     return result, left_fitx, right_fitx, ploty, vecicle_pos
    #
    # def measure_curvature_real(self, ploty, left_fitx, right_fitx, vehicle_pos):
    #     '''
    #     Calculates the curvature of polynomial functions in meters.
    #     '''
    #     # Define conversions in x and y from pixels space to meters
    #     ym_per_pix = 12 / 450  # meters per pixel in y dimension
    #     xm_per_pix = 3.7 / 910  # meters per pixel in x dimension
    #
    #     # Make sure to feed in your real data instead in your project!
    #     ploty_real = ploty * ym_per_pix
    #     left_fit_real = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    #     right_fit_real = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    #     vehicle_pos_real = vehicle_pos * xm_per_pix
    #     # Define y-value where we want radius of curvature
    #     # We'll choose the maximum y-value, corresponding to the bottom of the image
    #     y_eval = np.max(ploty_real)
    #
    #     # Calculation of R_curve (radius of curvature)
    #     left_curverad = ((1 + (
    #                 2 * left_fit_real[0] * y_eval * ym_per_pix + left_fit_real[1]) ** 2) ** 1.5) / np.absolute(
    #         2 * left_fit_real[0])
    #     right_curverad = ((1 + (
    #                 2 * right_fit_real[0] * y_eval * ym_per_pix + right_fit_real[1]) ** 2) ** 1.5) / np.absolute(
    #         2 * right_fit_real[0])
    #
    #     return left_curverad, right_curverad, vehicle_pos_real
    #
    # def line_tracker(self, line_para, new_value):
    #     '''
    #     line_tracker to track all the relevant line parameters,
    #     '''
    #     self.line_para.append(new_value)
    #
    # def current_is_outlier(self, line_para, current_para):
    #     '''
    #     Determine if current_para is an outlier when compared to previous line_para
    #     returns True if current_para is an outlier
    #     '''
    #     oulier_flag = False
    #     if len(line_para) == self.buffer_size:
    #         p = np.min(line_para)*0.8
    #         q = np.max(data)*1.2
    #         if current_para < q and current_para > p:
    #             return False
    #         else:
    #             return True
    #
    # def sanity_check(self, line_para, current_para):
    #     '''
    #     check if current detected line match with previous lines, and also are reasonable line
    #     '''
    #     flag_line = True
    #     # fit_plonomial() return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, vehicle_pos, leftx_base, rightx_base
    #     # first check the left and right line distance
    #     base_dist = current_para['leftx_base'] - current_para['rightx_base']
    #     line_dist = np.mean(current_para['left_fitx']) - np.mean(current_para['right_fitx'])
    #     if abs(base_dist-line_dist) > 50:
    #         flag_line = False
    #
    #     # second check whether the right and left line have similar curvature
    #     left_curverad, right_curverad, vehicle_pos_real = self.measure_curvature_real(current_para['ploty'],current_para['left_fitx'], current_para['right_fitx'], current_para['vehicle_pos'])
    #     if abs(left_curverad - right_curverad) > 50:
    #         flag_line = False
    #
    #     return flag_line
    #
    # def line_averaging(self, line_para):
    #     '''
    #     Average the coefficients stored in line_para
    #     '''
    #     return np.mean(self.line_para)