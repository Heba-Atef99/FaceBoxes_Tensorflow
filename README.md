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
    Pandas
    NumPy
    CV2
    matplotlib
    PIL
    tqdm
    flask
    flask_restful

## How to use the project.
### Pre-trained model
To use the pre-trained model, you need to download the frozen graph file (```model.pb```) from [here](https://drive.google.com/drive/folders/1D6vTNt6kiGT4pp6zI0C7-B0_fBw-Jcil?usp=sharing) and run ```api.py``` file (which depends on face_detector file) or use ```try_detector.ipynb``` notebook

### Evaluation
To evaluate the model using FDDB dataset go into ```Testing``` directory and:

  1. Download FDDB files from [here](https://drive.google.com/drive/folders/1Msy4RJS7aAqQng1VfjbNsPJ5JSkUkdfR?usp=sharing) into ```fddb``` folder
  2. Put the ```model.pb``` file in Testing directory
  3. Run ```explore_and_convert_FDDB.ipynb``` file to prepare the dataset to be ready for evaluation
  4. Run ```predict_for_FDDB.ipynb``` file to get the detections
  5. Go into ```eval_result``` and run ```FDDB.py``` to produce the discrete ROC & continous ROC files
  6. To visualize the FDDB annotations on the images run ```visulaize_original_annotations.ipynb```

### Training

For more details about the files dependencies and quick notes about each file, you can find it [here](https://drive.google.com/drive/folders/1BWwN0BDxybPw2crOg98ALnc4V3x-0FXY?usp=sharing) 
## Credits
This project is inspired by [this repo](https://github.com/TropComplique/FaceBoxes-tensorflow)
