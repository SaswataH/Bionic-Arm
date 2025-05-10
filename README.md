# Bionic Arm

An ML based hand gesture recognition model. Different hand poses are classified by the model implemented.

## Work flow

Step 1: Hand gesture recognition using Google's MediaPipe library.
Step 2: Collected data points are stored in a .csv file.
Step 3: The dataset is then fed to the ML Model.
Step 4: Live Hand gesture prediction using OpenCV live video capture.

## Present features

1. Support Vector Machine (with RBF kernel) model implemented
2. Recognises 3 hand gestures - Open Palm , Thumbs up , Closed fist with a 72% accuracy rate.

## Resources

[MediaPipe documentation](https://ai.google.dev/edge/mediapipe/solutions/guide)

[Wikipedia page on Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)

[Official SVM docs(Scikit-learn)](https://scikit-learn.org/stable/modules/svm.html)

[Official Open CV docs](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)

## Present problems

ML model stuck at approx. 70% accuracy. This often leads to failure to distinguish between Closed Fist and Thumbs Up.

## Immediate solutions

Collecting more varied datasets. Configuring SVM Model hyperparamters.

## Future goals

1. Collecting even more varied datasets for better training.
2. Implementing other ML models like Neural Network to compare with the present S.V.M model.
3. Passing serial data through a Microcontroller to control a 3-d printed Palm using I2C protocol.
