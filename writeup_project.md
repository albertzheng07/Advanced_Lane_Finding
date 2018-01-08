## Advanced Lane Finding Project Writeup

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

[image1]: ./examples/Undistorted_Result.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/Threshold_Result.jpg "Binary Example"
[image4]: ./examples/Warped_Result.jpg "Warp Example"
[image5]: ./examples/PolyFit_Result.jpg "Fit Visual"
[image6]: ./examples/Output_Result.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/project.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` function. In total, there were 20 test images that were used to calibrate the camera. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps occur in the 5th cell in the function `binary_threshold` in the Ipython notebook).

```python
def binary_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel with G and B channels corresponding to x gradient and saturation channel results
    zero_channel = np.zeros_like(sxbinary)
    color_binary = np.dstack((zero_channel, sxbinary, s_binary)) * 255

    # Take union of x gradient and saturation channel as threshold mask
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary
```


I first converted the image into the `HLS (Hue Light Saturation)` color space in order to re-characterize the image for the threshold operations.  

I used the `cv2.Sobel()` function to take the derivative in the x direction of the image in the L channel. I took a threshold gradient between 20 and 100 which would capture edges in the vertical direction. Then, I did a direct threshold on the S channel looking for high amount of saturation which would indicate dense color that would correlate with a line. I combined these thresholds by intersected them. The final combined result and individually stacked results can be seen in the example below.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 7th code cell of
the IPython notebook.

```python
def warper(img, src, dst):

    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped
```

The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 70, img_size[1] / 2 + 118],
    [((img_size[0] / 6) + 30), img_size[1]],
    [(img_size[0] * 5 / 6) + 100, img_size[1]],
    [(img_size[0] / 2 + 120), img_size[1] / 2 + 118]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 570, 478      | 320, 0        |
| 243.3, 720      | 320, 720      |
| 116.7, 720     | 960, 720      |
| 760, 478      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

```python
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):

  window_centroids = [] # Store the (left,right) window centroid positions per level
  window = np.ones(window_width) # Create our window template that we will use for convolutions

  # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
  # and then np.convolve the vertical image slice with the window template

  # Sum quarter bottom of image to get slice, could use a different ratio
  l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
  #print(l_sum.shape)
  #print(window.shape)

  l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
  r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
  r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

  # Add what we found for the first layer
  window_centroids.append((l_center,r_center))

  # Go through each layer looking for max pixel locations
  for level in range(1,(int)(image.shape[0]/window_height)):
      # convolve the window into the vertical slice of the image
      image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
      conv_signal = np.convolve(window, image_layer)
      # Find the best left centroid by using past left center as a reference
      # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
      offset = window_width/2
      l_min_index = int(max(l_center+offset-margin,0))
      l_max_index = int(min(l_center+offset+margin,image.shape[1]))
      l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
      # Find the best right centroid by using past right center as a reference
      r_min_index = int(max(r_center+offset-margin,0))
      r_max_index = int(min(r_center+offset+margin,image.shape[1]))
      r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
      # Add what we found for that layer
      window_centroids.append((l_center,r_center))

  return window_centroids
```

```python
def find_left_right_points(image, window_centroids, window_width, window_height):   
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
           # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,image,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,image,window_centroids[level][1],level)
           # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((image, image, image))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((image,image,image)),np.uint8)

    return output, l_points, r_points
```

```python
def fitpolypixels(l_points, r_points):
    # I get the pixel locations of each lane line
    left = np.argwhere(l_points == 255) # y,x
    right = np.argwhere(r_points == 255)

    # Fit a second order polynomial to pixel positions in each lane line
    left_fit = np.polyfit(left[:,0], left[:,1], 2) # y,x
    left_fitx = left_fit[0]*left[:,0]**2 + left_fit[1]*left[:,0] + left_fit[2]
    right_fit = np.polyfit(right[:,0], right[:,1], 2)
    right_fitx = right_fit[0]*right[:,0]**2 + right_fit[1]*right[:,0] + right_fit[2]

    return left, right, left_fitx, right_fitx, left_fit, right_fit            
```

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

```python
def get3pointRadius(x1,y1,x2,y2,x3,y3):
    m1 = (y2-y1)/(x2-x1)
    m2 = (y3-y2)/(x3-x2)
    xc = (m1*m2*(y1-y3)+m2*(x1+x2)-m1*(x2+x3))/(2*(m2-m1))
    yc = -(xc-(x1+x2)/2)/m1 +(y1+y2)/2      

    radius = np.sqrt((x2-xc)**2+(y2-yc)**2)

    return radius

def calcRadiusCurv(left, left_fit, right_fit):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_fit_cr = np.polyfit(left[:,0]*ym_per_pix, left[:,1]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(right[:,0]*ym_per_pix, right[:,1]*xm_per_pix, 2)    

    y_min = np.min(left[:,0]*ym_per_pix) # min y
    y_mean = np.mean(left[:,0]*ym_per_pix) # mean y
    y_max = np.max(left[:,0]*ym_per_pix) # max y

    leftx_min = np.polyval(left_fit_cr,y_min)
    leftx_mean = np.polyval(left_fit_cr,y_mean)
    leftx_max = np.polyval(left_fit_cr,y_max)
    rightx_min = np.polyval(right_fit_cr,y_min)
    rightx_mean = np.polyval(right_fit_cr,y_mean)
    rightx_max = np.polyval(right_fit_cr,y_max)

    # calculate the curvature with 3 min, mean and max points of the image
    left_curverad = get3pointRadius(leftx_min,y_min,leftx_mean,y_mean,leftx_max,y_max)
    right_curverad = get3pointRadius(rightx_min,y_min,rightx_mean,y_mean,rightx_max,y_max)

    return left_curverad, right_curverad

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `drawPolygon()`.  

``` python
def drawPolygon(warp_image, image, undist, dst, src, leftx, rightx, left_curverad, right_curverad):
    # Draw on Polygon back onto original image

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, left[:,0]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right[:,0]])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    Mi = cv2.getPerspectiveTransform(dst, src)

    left_mean = np.mean(leftx) # get avg of left lane pixels
    right_mean = np.mean(rightx) # get avg of right lane pixels
    vehicle_pos = (image.shape[1]/2)-np.mean([left_mean, right_mean]) # assume camera pos is center of image. vehicle pos w.r.t. vehicle lane
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    avg_curv = (left_curverad+right_curverad)/2    

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Mi, (image.shape[1], image.shape[0]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1.2
    fontColor = (255,255,255) # white
    thickness = 4
    cv2.putText(newwarp,'Vehicle Position' + ' is ' + str(vehicle_pos*xm_per_pix)[:5] + ' m Left of Center',(10,30), font, fontscale, fontColor,thickness)
    cv2.putText(newwarp,'Radius of Curvature = ' +str(avg_curv)[:5] + ' m' ,(10,80), font, fontscale,fontColor,thickness)

    # Combine the result with the original image
    undist = cv2.undistort(image, mtx, dist, None, mtx) # undistort with camera calibration   
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result
```
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
