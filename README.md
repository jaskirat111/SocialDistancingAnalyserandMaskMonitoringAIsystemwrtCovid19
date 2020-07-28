# Live Social Distance Monitoring and Face Mask Detection AI Solution for Covid-19

## Tata Innoverse Solver Hunt 8 Hackathon

### This project proposes a computer vision based AI system to check whether social distancing is being maintained in crowded place or at any place (eg: market, or workplace) coupled with Mask detection system to track people who are wearing masks. This solution can be used in CCTV cameras and other video surveillance systems. While the data; such as the number of people in the vicinity, the number of people violating social distancing, and not wearing face-masks; has been used for analysis.

### Results

 ![](Result.gif)
 
As we can see, the system identifies the people in the frame and puts light green, dark red or orange bounding boxes if they are safe (No risk), at High risk or at low risk level respectively. The connecting lines among the persons shows the level of closenes among people (Red shows very close and yellow shows close). After detecting the person, the system also detects the face and identify whether the person is masked or not by putting Green or Red bounding box. Status is shown in a bar at the bottom, showing all the details.These are pretty decent results considering the complexity of the problem we have at our hands.
 
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
* [PIL](https://pillow.readthedocs.io/en/stable/) : Used for manipulating images.
* [OpenCV](https://opencv.org/) : Used for manipulating images and video streams.
* [Keras](https://keras.io/) : Used for designing and training the Face_Mask_Classifier model.
* [face-detection](https://github.com/hukkelas/DSFD-Pytorch-Inference) : Used for detecting faces with Dual Shot Face Detector.
* [tqdm](https://github.com/tqdm/tqdm) : Used for showing progress bars.
* [Google Colab](https://colab.research.google.com/) : Used as the developement environment for executing high-end computations on its backend GPUs/TPUs and for editing Jupyter Notebooks. 
* [imutils](https://pypi.org/project/imutils/) : Used for resizing each frame to (1280,720)

## Project Description and Results
### Person Detection
[YOLO](https://pjreddie.com/darknet/yolo/) (You Only Look Once) is a state-of-the-art, real-time object detection system. It's Version 3 based spp model(pretrained on COCO dataset), with a resolution of 608x608 in used in this project for obtaining the bounding boxes of individual persons in a video frame. To obtain a faster processing speed, a resolution of 416x416 or 320x320 can be used. YOLOv3-tiny can also be used for speed optimization. However it will result in decreased detection accuracy.


### Face Detection
[Dual Shot Face Detector](https://github.com/Tencent/FaceDetection-DSFD) (DSFD) is used throughout the project for detecting faces. Common Face Detectors such as the Haar-Cascades or the MTCNN are not efficient in this particular use-case as they are not able to detect faces that are covered or have low-resolution. DSFD is also good in detecting faces in wide range of orientations. It is bit heavy on the pipeline, but produces accurate results.

### Face Mask Classifier
A slighly modified ResNet50 model (with base layers pretrained on imagenet) is used for classifying whether a face is masked properly or not. Combination of some AveragePooling2D and Dense (with dropout) layers ending with a Sigmoid or Softmax classifier is appended on top of the base layers. Different architectures can be used for the purpose, however complex ones should be avoided to reduce overfitting. The model needs to be trained on tons of relevant data before we can apply it in real-time and expect it to work. It needs a lot of computational power and I mean a lot! We can try our models trained on a small dataset in our local machines, but it would not produce desirable results. Therefore I used pretrained open-source models for now. So we use the model trained by the team of [Thang Pham](https://github.com/aome510/Mask-Classifier) for this purpose. It is basically a ResNet50 Model with a modified top.

Implementation details can be found in this [notebook](https://github.com/jaskirat111/Social-Distancing-Analyser-and-Mask-Monitoring-AI-system-wrt-Covid-19/blob/master/Social_Distancing_Monitor_Face_mask_Detection.ipynb). 


### Future Scope
Some optimizations can be made in the form of vectorization. For getting the position of a person, there are various approaches. One of them being simply using the centers of the bounding boxes, the one used in this project. Other one is using OpenCV's perspective transform to get a bird's eye view of the positions, but that kind of needs pretty accurate frame of reference points. Using it also increases the complexity of the system by a bit. However if implemented correctly, it will no doubt produce better results. For now we stick to the first approach. Remember, there's always scope for improvements!


