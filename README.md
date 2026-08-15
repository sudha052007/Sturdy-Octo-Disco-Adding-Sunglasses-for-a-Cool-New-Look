# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!


## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## Program
~~~
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread('Photo.jpeg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
~~~
<img width="538" height="683" alt="image" src="https://github.com/user-attachments/assets/6b07d6b9-07b6-40f6-a94c-54636d5972ec" />


~~~
faceImage.shape
glassPNG = cv2.imread('glass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
~~~
<img width="845" height="462" alt="image" src="https://github.com/user-attachments/assets/364caa4d-d670-4ca6-a1e9-7db8a66daed2" />

~~~
glassPNG = cv2.resize(glassPNG, (185,65))
print("image Dimension ={}".format(glassPNG.shape))

glassBGR = glassPNG[:, :, :3]
glassMask1 = glassPNG[:, :, 3]

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
~~~
<img width="917" height="208" alt="image" src="https://github.com/user-attachments/assets/805deb59-90f3-46e2-a2df-61b5ba0dead7" />

~~~
# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[150:215,135:320] = glassBGR


plt.imshow(glassPNG[:, :, ::-1])
~~~
<img width="483" height="194" alt="image" src="https://github.com/user-attachments/assets/46a84328-702d-491f-a942-6c810491b7e6" />

~~~
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))
# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)
# Make a copy
faceWithGlassesArithmetic = faceImage.copy()
# Get the eye region from the face image
eyeROI = faceWithGlassesArithmetic[150:215,135:320]
# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))
# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)
# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)
# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")

~~~
<img width="2164" height="727" alt="image" src="https://github.com/user-attachments/assets/0b09928e-7376-4c03-9fee-abc9f8dd34cb" />

~~~
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[150:215,135:320]=eyeRoiFinal
# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
~~~
<img width="1630" height="965" alt="image" src="https://github.com/user-attachments/assets/74bc137a-140b-4d81-9cb5-ac5d5fd22f86" />


## Result:
The sunglasses PNG image was successfully superimposed onto the face image using alpha masking and image blending techniques in OpenCV. The final output shows a realistic face image with sunglasses correctly placed over the eyes.
