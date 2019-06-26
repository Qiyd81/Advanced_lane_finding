#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline
#reading in an image
from imageio.plugins import ffmpeg

image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


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


### Below is my simple version code which works, but result not so good.
def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # for line in lines
    # for x1,y1,x2,y2 in line:
    # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    ### Below is my code, not finished yet
    imshape = img.shape
    slope_left = 0
    slope_right = 0
    left_x = 0
    left_y = 0
    right_x = 0
    right_y = 0
    i = 0
    j = 0
    #   slopes_left = []       # list used to remove noise, will be deleted if doesn't work
    #  slopes_right = []     # list used to remove noise, will be deleted if doesn't work

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0.2:
                slope_left += slope
                left_x += (x1 + x2) / 2
                left_y += (y1 + y2) / 2
                i += 1
                # slopes_left.append(slope)  # list used to remove noise, will be deleted if doesn't work
            elif slope < -0.2:
                slope_right += slope
                right_x += (x1 + x2) / 2
                right_y += (y1 + y2) / 2
                j += 1
    # slopes_right.append(slope)  # list used to remove noise, will be deleted if doesn't work
    # slopes_left, removed_index_left = remove_noise(slopes_left)   # list used to remove noise, will be deleted if doesn't work
    # slopes_right, removed_index_right = remove_noise(slopes_right)   # list used to remove noise, will be deleted if doesn't work
    y_bottom = np.int(0.95 * imshape[0])
    y_top = np.int(0.6 * imshape[0])
    if i > 0:
        avg_slope_left = slope_left / i
        # avg_slope_left = np.mean(slopes_left) # list used to remove noise, will be deleted if doesn't work
        avg_left_x = left_x / i
        avg_left_y = left_y / i

        left_x_bottom = np.int((y_bottom - avg_left_y) / avg_slope_left + avg_left_x)

        left_x_top = np.int((y_top - avg_left_y) / avg_slope_left + avg_left_x)

    else:  # Just assume one line
        left_x_bottom = np.int(0.2 * imshape[1])
        left_x_top = np.int(0.4 * imshape[1])
    cv2.line(img, (left_x_bottom, y_bottom), (left_x_top, y_top), color, thickness)

    if j > 0:
        avg_slope_right = slope_right / j
        # avg_slope_right = np.mean(slopes_right)# list used to remove noise, will be deleted if doesn't work
        avg_right_x = right_x / j
        avg_right_y = right_y / j

        right_x_bottom = np.int((y_bottom - avg_right_y) / avg_slope_right + avg_right_x)
        right_x_top = np.int((y_top - avg_right_y) / avg_slope_right + avg_right_x)
    else:  # Just assume one line
        right_x_bottom = np.int(0.8 * imshape[1])
        right_x_top = np.int(0.6 * imshape[1])

    # Draw a line
    cv2.line(img, (right_x_bottom, y_bottom), (right_x_top, y_top), color, thickness)


### Above is my simple version code, which works, but the result not so good.

###Below are my functions to test other solution, not succeed yet.
# def cal_slopes_bias(lines):
#	slopes_left = np.zeros(len(lines))
#	slopes_right = np.zeros(len(lines))
#	b_left = np.zeros(len(lines))
#	b_right = np.zeros(len(lines))
#	for line in lines:
#		line_index = map(tuple, lines).index(line)
#		lines_left = []
#		lines_right = []
#		for x1, y1, x2, y2 in line:
#			m_line, b_line = np.polyfit((x1, y1), (x2, y2), 1)
#			if m_line<0 and x1<x_size/2 and x2<x_size/2:
#				slopes_left[line_index] = m_line
#				b_left[line_index] =  b_line
#				lines_left.append(lines[line_index])
#			elif m_line>0 and x1>x_size/2 and x2>x_size/2:
#				slopes_right[line_index] = m_line
#				b_right[line_index] = b_line
#				lines_right.append(lines[line_index])
#	return slopes_left, b_left, slopes_right, b_right, lines_left, lines_right
# def cal_slopes_mean(slopes):
#	if slopes!=[]:
#		slopes_mean = np.mean(slopes)
#	return slopes_mean
##def find_deviation_line(lines):
##	slopes_left, b_left, slopes_right, b_right = cal_slopes(lines)
##	slopes_left_mean,slopes_left_standard_deviation = cal_slopes_mean_deviation(slopes_left)
##	b_left_mean,b_left_standard_deviation = cal_slopes_mean_deviation(b_left)
##	slopes_right_mean,slopes_right_standard_deviation = cal_slopes_mean_deviation(slopes_right)
##	b_right_mean,b_right_standard_deviation = cal_slopes_mean_deviation(b_right)

# def noise_index(x, m=1.01):
#	mean_value = np.mean(x)
#	standard_deviation = np.std(x)
#	removed_index = []
#	for num in x:
#		if num!=0 and abs(num - mean_value) > (m*standard_deviation):
#			removed_index.append(x.index(num))
#	return removed_index
#	# whether or not need? return b_left, left_line_x, left_line_y
# def mean_x_y(lines):
#	x1 = 0
#	y1 = 0
#	x2 = 0
#	y2 = 0
#	i = 0
#	for line in lines:
#		for x1, y1, x2, y2 in line:
#			x1 += x1
#			y1 += y1
#			x2 += x2
#			y2 +=y2
#			i += 1
#		x_mean = (x1 + x2)/(2*i)
#		y_mean = (y1 + y2)/(2*i)
#	return x_mean, y_mean
# def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
#	slopes_left, b_left, slopes_right, b_right, lines_left, lines_right = cal_slopes_bias(lines)
#	removed_index_left = noise_index(slopes_left)
#	removed_index_left.append(noise_index(b_left))
#	removed_index_right = noise_index(slopes_right)
#	removed_index_right.append(noise_index(b_right))
#	removed_index_all = removed_index_left.append(removed_index_right)
#
#	lines_left.remove[removed_index_left]
#	lines_right.remove[removed_index_right]
#	lines.remove[removed_index_all]
#
#	slopes_left, b_left, slopes_right, b_right, lines_left, lines_right = cal_slopes_bias(lines)
#
#	slopes_left_mean = cal_slopes_mean(slopes_left)
#	slopes_right_mean = cal_slopes_mean(slopes_right)
#
#	x_mean_left, y_mean_left = mean_x_y(lines_left)
#	x_mean_right, y_mean_right = mean_x_y(lines_right)
#	y1_output = y_size
#	x1_output_left = np.int((y1_output - y_mean_left)/slopes_left_mean + x_mean_left)
#	y2_output = np.int(y_size*0.6)
#	x2_output_left = np.int((y2_output - y_mean_left)/slopes_left_mean + x_mean_left)
#	x1_output_right = np.int((y1_output - y_mean_right)/slopes_right_mean + x_mean_right)
#	x2_output_right = np.int((y2_output - y_mean_right)/slopes_right_mean + x_mean_right)
#
#	cv2.line(img, (x1_output_left, y1_output), (x2_output_left, y2_output), color, thickness)
#	cv2.line(img, (x1_output_right, y1_output), (x2_output_right, y2_output), color, thickness)
###	### above is my optimize code, not finished yet.

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

import os
os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
dir = os.listdir("test_images/")
print(dir)
for imgi in dir:
    image = mpimg.imread("test_images/"+imgi)
    image_gray = grayscale(image)
    image_canny = canny(image_gray, 50, 150)
    plt.imshow(image_canny)
    image_gaussian = gaussian_blur(image_canny, 5)
    plt.imshow(image_gaussian)
    ysize = image_gaussian.shape[0]
    print(ysize)
    xsize = image_gaussian.shape[1]
    print(xsize)
    vertices = [np.array([[np.int(xsize*0.1), ysize], [np.int(xsize*0.45), np.int(ysize*0.6)], [np.int(xsize*0.6), np.int(ysize*0.6)], [xsize, ysize]])]
    print(vertices)
    image_region = region_of_interest(image_gaussian, vertices)
    plt.imshow(image_region)
    image_hough = hough_lines(image_region, 1, np.pi/180, 15, 40, 10)
    plt.imshow(image_hough)
    image_output = weighted_img(image_hough, image)
    plt.imshow(image_output)
    cv2.imwrite("../test_image_output/"+imgi, image_output)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#imageio.ffmpeg.download() has been deprecated. Use 'pip install imageio-ffmpeg' instead.' And import it separately.


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    image_gray = grayscale(image)
    image_canny = canny(image_gray, 100, 150)
    image_gaussian = gaussian_blur(image_canny, 5)
    ysize = image_gaussian.shape[0]
    xsize = image_gaussian.shape[1]
    vertices = [np.array([[np.int(xsize*0.1), np.int(ysize*0.9)], [np.int(xsize*0.45), np.int(ysize*0.6)], [np.int(xsize*0.6), np.int(ysize*0.6)], [xsize, np.int(ysize*0.9)]])]
    image_region = region_of_interest(image_gaussian, vertices)
    image_hough = hough_lines(image_region, 10, np.pi/180, 10, 40, 10)
    result = weighted_img(image_hough, image)
    return result



white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
#%time yellow_clip.write_videofile(yellow_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
#%time challenge_clip.write_videofile(challenge_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))