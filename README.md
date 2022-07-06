# FaceBoxes Tensorflow

<div align="center">
  <img height="200" src="assets/face_detection.png">
</div>

## Description

This project is an implementation of this [paper](https://arxiv.org/abs/1708.05234) with the help of this [repo](https://github.com/TropComplique/FaceBoxes-tensorflow) which proposes a solution that compines the speed and accuracy of performance for the state-of-art problem of face detection.

It is part of [Smart Exam Website](https://github.com/Smart-Exam-Website) project in which it serves the feature of detecting the faces of the present individuals in front of the camera and returns the number of faces detected to indicate if there is someone beside the student helping in cheating in the exam or not.


## Table of Contents.
## Dependencies:

    Python 3.9.7
    Tensorflow 2.8.0 (but the part of saving model and creating pb file needs Tesnsorflow 1.12)
    NumPy
    CV2
    matplotlib
    PIL
    tqdm
    flask
    flask_restful

## How to use the project.
### Pre-trained model
To use the pre-trained model, you need to download the frozen graph file (model.pb) from [here](https://drive.google.com/drive/folders/1D6vTNt6kiGT4pp6zI0C7-B0_fBw-Jcil?usp=sharing) and run the API file (which depends on face_detector file) or use try_detector notebook

### Evaluation
### Training
## Credits
This project is inspired by [this repo](https://github.com/TropComplique/FaceBoxes-tensorflow)
