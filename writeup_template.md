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

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

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

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

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

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
