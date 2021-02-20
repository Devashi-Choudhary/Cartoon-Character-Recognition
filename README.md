# Cartoon-Character-Recognition
Cartoon Character one of the people or animals in an animated film like Mickey Mouse, Tom & Jerry etc. Cartoons are essential part of every childhood. They are, certainly, the most popular entertainment for children, but also much more than that. With the help of cartoons kids can learn about the world around us, about new emotions, life issues and other important things. Hence, just for fun the goal of current project is to recognize the cartoon character using deep learning algorithm. 

# Dependencies
Deep Learning based Cartoon Character Recognition uses [OpenCV](https://opencv.org/) (opencv==4.2.0) and Python (python==3.7). The model Convolution Neural Network(CNN) uses [Keras](https://keras.io/) (keras==2.3.1) on [Tensorflow](https://www.tensorflow.org/) (tensorflow>=1.15.2). Also, imutils==0.5.3, numpy==1.18.2, matplotlib==3.2.1, argparse==1.1 are also used.

# How to execute code:

1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. Now, install the libraries required.
4. Now, you can create your own dataset and put it in the `Data/` folder in following format
  `Data/Category1`
  `Data/Category2` and so on.
5. **Training of CNN Model :** You can check `Training.ipynb` for training and save the trained model inside `model/` folder.
6. **Testing of CNN Model :**  You can use pretrained model and run the following command :

For recognizing cartoon character in images, run the following command :
> `python Cartoon_Character_Recognition_in Image.py --path Data/Donald.jpeg --model Model/model.h5`

For detecting face mask in real-time video stream, run the following command :
> `python Cartoon_Character_Recognition_in Video.py --path Data/video.mp4 --model Model/model.h5`


# Results

1. Accuracy/Loss training curve plot.

![Accuracy and Loss](https://github.com/Devashi-Choudhary/Cartoon-Character-Recognition/blob/main/Results/download.png)


2. Cartoon Character Recognition in Image.

![predicted](https://github.com/Devashi-Choudhary/Cartoon-Character-Recognition/blob/main/Results/Output_test.JPG)

3. Cartoon Character Recognition in Video.



**NOTE :** For more information about implementation details of real-time creativity, please go through [Recognizing Real-Time Creativity of User using Deep Learning
](https://medium.com/@Devashi_Choudhary/recognizing-real-time-creativity-of-user-using-deep-learning-786cbc5cd292)


# Note 
I have created my own Dataset. I have collected the Data from Chrome and various sites like [Disney](https://www.disneyclips.com/images/donaldduck9.html). As of now, only 4 categories of cartoon characters (Mickey-Mouse, Donald-Duck, Minions, Pooh) are considered. You can create your own dataset and include more categories and save the data in `.npy` format. If you want to use my Dataset and pretrained model, it will come with little cost as it requires so much time to collect tha data and all preprocessing. Please drop a mail on devashi882@gmail.com for accessing the data and pretrained model.
