##Alphabet Hand Gesture Recognition using Mediapipe

Mediapipe is a cross-platform framework which is used to build many Machine Learning pipelines for Hands, Object Detection, Face Mesh, etc.

This repository consists of following contents.

* Alphabet Gestures Data collection and Training
* App.py for Testing the model

**Requirements**

* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
*	TensorFlow 2.3.0 or Later
*	tf-nightly 2.5.0.dev or later
*	scikit-learn 0.23.2 or Later
*	matplotlib 3.3.2 or Later

**Demo**

To run the demo using your webcam run python app.py 

**Following options are used to run the demo**
*	--device -> Specifying the Camera Device Number (default=0)
*	--width -> Width at the time of camera capture (default=960)
*	--height -> Height at the time of camera capture (default=540)
*	--use_static_image_mode -> Whether to use static_image_mode option for Media Pipe inference (Default：Unspecified)
*	--min_detection_confidence -> Detection confidence threshold (default:0.5)
*	--min_tracking_confidence -> Tracking confidence threshold (default：0.5)

**Dataset**

![image](https://user-images.githubusercontent.com/37393700/114131200-ce5a6a80-991f-11eb-91a8-a8e51afdf2e7.png)

**Data Collection**

1.	To collect data press key "k" while running the app.py file which switches to listening mode as displayed in the figure below
2.	Then by pressing keys from 0 to 9, we can load each gesture for the hand gesture label. 
3.	After Finding coordinates, Key Point Classifier undergoes 4 steps of Data Pre-processing steps namely, Landmark Coordinates, relative coordinate conversion, Flattening to     a 1-D array, and Normalized values.
4.	Then the key points will be then added to “model/keypoint_classifier/keypoint.csv” as shown below. 

![image](https://user-images.githubusercontent.com/37393700/114130490-4e7fd080-991e-11eb-9564-d733649bcaa3.png) 

5.	1st column denotes pressed number (used as class ID), 2nd and subsequent columns- Keypoint coordinates
6.	As we are training 26 Alphabets we required to record and save all 26 labels. For this we undergo, 10 labels + 10 labels + 6 labels respectively.

**KeyPoint Coordinates**

![image](https://user-images.githubusercontent.com/37393700/114130509-56d80b80-991e-11eb-816c-fbc5fe61555b.png)
 
**Pre-Processing Steps**

![image](https://user-images.githubusercontent.com/37393700/114130521-5f304680-991e-11eb-8485-1ffd59f50a5e.png)

**Model Training**

The model structure for training the key points can be found in “alpha_train.ipynb" in Jupyter Notebook and execute from top to bottom. We used 75% of the data for training and the rest 25% is allocated for testing purpose. 

**Model Architecture**

This model architecture with the following layers shown below has been used for training in alpha_train.ipynb

![image](https://user-images.githubusercontent.com/37393700/114130544-6f482600-991e-11eb-98b6-59e113559787.png)
 
**References**

*	https://mediapipe.dev/
*	https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe

**Authors**

*	Balaji 
*	Padmajaya Rekha K
*	Siddhartha N
