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
[image5]: ./examples/New_PolyFit_Result.jpg "Fit Visual"
[image6]: ./examples/Output_Result.jpg "Output"
[video1]: ./project_video_out.mp4 "Video"

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
def binary_threshold(img, s_thresh=(180, 255), l_thresh=(225, 255), b_thresh=(155, 200)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    l_channel = luv[:,:,0]

    #lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    #b_channel = #lab[:,:,2]
    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2]   

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x on gray scale of image
    #scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))    
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # Take the derivative in y on gray scale of image
    gradmag = np.sqrt(sobelx**2 + sobely**2)
   # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Threshold light channel to be more robust to shadows
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1   

    # Threshold yellow colors
    b_binary = np.zeros_like(b_channel)

    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1  

    # Threshold saturation/color channel to detect both yellow/white lines
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    combined_binary = np.zeros_like(b_binary)
    combined_binary[(b_binary == 1) | (l_binary == 1)] = 1

    return combined_binary
```


I first converted the image into the `HLS (Hue Light Saturation)`, `LAB` and `LUV` color spaces in order to re-characterize the image for the threshold operations. I found it sufficient to combine the light channel and blue channel to capture the yellow and white lines.

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

I used a sliding window method which identified the nonzero pixels in the image that
captured from the binary threshold. I then fit the captured pixel positions to
a polynomial to come up with lane lines.

```python
def sliding_window(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 75
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    isLeftLineDetected = (leftx.size > 0 and lefty.size > 0)
    isRightLineDetected = (righty.size > 0 and rightx.size > 0)    

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left = np.zeros((len(left_lane_inds),2))
    right = np.zeros((len(right_lane_inds),2))

    left[:,1] = leftx
    left[:,0] = lefty

    right[:,1] = rightx
    right[:,0] = righty    

    return left, right, left_fitx, right_fitx, ploty, left_fit, right_fit, isLeftLineDetected, isRightLineDetected, out_img
```

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In order to calculate the radius curvature of the lane, I calculated the radius using 3 points of each lane. I picked the min, max and mean y points along each line and implemented the formula from the following reference (https://www.intmath.com/applications-differentiation/8-radius-curvature.php). In order to scale the values to real world values, I converted the pixels to meters and did a poly fit on the scaled meter values in order to get the 3 points in real world values. Then, I computed the radius of curvature for both lane lines. This can be seen in the code cells 11 and 12.

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
I computed the vehicle's position with respect to the center of the lane line by assuming the center of the camera was on the middle of the vehicle. With the assumption, you can subtract the middle of the two lanes from the middle of the image to get the relative position in pixels. You then convert that into meters to get the final result.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

For the final step, I took the lane lines that were fitted and drew the polygon onto the undistorted image in function `drawPolygon()`. I added the text to identify the average radius of curvature between the two lane lines and the vehicle's relative position from the center of the lanes. This can be seen in code cell 13.    

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

I found the simple lane finder was pretty sensitive to the margins surrounding the previously found lines. I had to shrink the margin values to avoid identifying objects in other lanes i.e. other cars. I also had to tune the binary thresholds and when encountering different backgrounds such as the lighter road on the bridge. A more adaptive filter to be robust to different backgrounds versus standard roads would have been useful or different thresholds for that specific background. I found myself trying a one sized fits all solution.

The pipeline could also fail if the lane lines change drastically in position (narrower or wider all of a sudden) where the perspective transform would no longer capture the correct source points. If there was a lower weight algorithm with high enough confidence to dynamically calculate the source points, this would provide more robustness to the pipeline.
