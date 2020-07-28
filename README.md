# Live Social Distance Monitoring and Face Mask Detection AI Solution for Covid-19

## Tata Innoverse Solver Hunt 8 Hackathon

### This project proposes a computer vision based AI system to check whether social distancing is being maintained in crowded place or at any place (eg: market, or workplace) coupled with Mask detection system to track people who are wearing masks. This solution can be used in CCTV cameras and other video surveillance systems. While the data; such as the number of people in the vicinity, the number of people violating social distancing, and not wearing face-masks; has been used for analysis.




 ![](Result.gif)
 
 ![](Result1.gif)

  ![](Result2.gif)
  
  ![](Result4.gif)

  ![](Result3.gif)

  ![](Result5.gif)

  ![](Result6.gif)
  
  ![](Result7.gif)
  
  ![](Result8.gif)

# Project Overview
Our AI Compliance consists of state-of-the-Art Social Distancing Monitoring coupled with Mask detection system to check whether the norms are followed or not. This project aims at monitoring people violating Social Distancing over video footage coming from CCTV Cameras. Uses YOLOv3-spp for detecting persons along with social distancing analyser tool simulated with 3D depth factor based on the camera position and orientation for recognizing potential intruders. Dual Shot Face Detector s used for detecting faces and A Face Mask Classifier model (ResNet50) is trained and deployed for identifying people not wearing a face mask.

A detailed description of this project along with the results can be found [here](#project-description-and-results).

## Getting Started

### Prerequisites
Running this project on your local system requires the following packages to be installed :

* numpy
* matplotlib
* sklearn
* PIL
* cv2
* keras 
* face_detection
* tqdm
* imutils

They can be installed from the Python Package Index using pip as follows :
 
     pip install numpy
     pip install matplotlib
     pip install sklearn
     pip install Pillow
     pip install opencv-python
     pip install Keras
     pip install face-detection
     pip install tqdm
     pip install imutils
     
You can also use [Google Colab](https://colab.research.google.com/) in a Web Browser with most of the libraries preinstalled.

### Usage
This project is implemented using interactive Jupyter Notebooks. You just need to open the notebook on your local system or on [Google Colab](https://colab.research.google.com/) and execute the code cells in sequential order. The function of each code cell is properly explained with the help of comments.

Please download the following files (from the given links) and place them in the Models folder in the root directory :
1. YOLOv3 spp weights :  https://pjreddie.com/media/files/yolov3-spp.weights
2. Face Mask Classifier ResNet50 Keras Model : https://drive.google.com/file/d/1Bf7hH5Ugu2B7gA6aVpi7xQD1cRpIPOUm/view?usp=sharing

Also before starting you need to make sure that the path to various files and folders in the notebook are updated according to your working environment. If you are using [Google Colab](https://colab.research.google.com/), then :
1. Mount Google Drive using : 

        from google.colab import drive
        drive.mount("/content/drive/My Drive")
        
2. Update file/folder locations as `"path_to_file_or_folder"`.

## Tools Used
* [NumPy](https://numpy.org/) : Used for storing and manipulating high dimensional arrays.
* [Matplotlib](https://matplotlib.org/) : Used for plotting.
* [Scikit-Learn](https://scikit-learn.org/stable/) : Used for DBSCAN clustering.
* [PIL](https://pillow.readthedocs.io/en/stable/) : Used for manipulating images.
* [OpenCV](https://opencv.org/) : Used for manipulating images and video streams.
* [Keras](https://keras.io/) : Used for designing and training the Face_Mask_Classifier model.
* [face-detection](https://github.com/hukkelas/DSFD-Pytorch-Inference) : Used for detecting faces with Dual Shot Face Detector.
* [tqdm](https://github.com/tqdm/tqdm) : Used for showing progress bars.
* [Google Colab](https://colab.research.google.com/) : Used as the developement environment for executing high-end computations on its backend GPUs/TPUs and for editing Jupyter Notebooks. 
* [imutils](https://pypi.org/project/imutils/) : Used for resizing each frame to (1280,720)




