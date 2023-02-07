# COVID-19-CLASSIFICATION-FROM-CHEST-X-RAY-IMAGES

covid-19 classification from chest x ray images using convolutional neural networks


**INTRODUCTION:**

The  Coronavirus disease caused by SARS, in response to the rapidly increasing number of cases of
the emerging disease, the drawbacks of manual testing include the availability of testing kits, which are costly and inefficient
blood tests; a blood test takes hours to generate the result.
So, the idea is to overcome these limitations using the Deep Learning technique for
efficient treatment. The faster we produce the results, the fewer cases in the city,
that’s why we can use CNN to get our job done

**DATASET:**

COVID-19 chest x-ray image dataset: Dataset of chest X-ray and CT images of patients which are positive or suspected of COVID-19.Total of 930 images of all diseases scans.
Normal person chest x-ray image dataset:There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).1341 images are of Normal category.

**OBJECTIVES:**

Develop a cnn model adapted from VGG-16.

**PREPROCESSING:**

Converting 224x224 RGB image to BGR image, because OpenCV library uses
BGR image format.
Scaling down pixel values to 0 and 1 by dividing with 255
Data augmentation: Horizontal flip, Slight zoom, Slight shear.
Resize to 224x224x3 RGB to BGR format.
Splitting of dataset into train, validation, test 8:1:1

**MODEL ARCHITECTURE:**





![Screenshot_20230207_102339](https://user-images.githubusercontent.com/56174010/217151845-d96d552a-991b-41dd-9618-66e2dbffc93a.png)


**RESULTS:**

![Screenshot_20230207_102349](https://user-images.githubusercontent.com/56174010/217151894-3391b738-9561-43d6-b657-f13412f55067.png)

