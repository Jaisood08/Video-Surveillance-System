Hi this is a Opencv project with deep learning . 

In deep learning for face recognition i have used deep metric learning where instead of trying to output
a single label (or even the coordinates/bounding box of objects in an image), i am instead outputting 
a real-valued feature vector that is used to quantify the face.

Training the network is done using triplets. Here, we need to provide three images to the network:
   o   Two of these images are example faces of the same person.
   o   The third image is a random face from our dataset and is not the same person as the other two images.

Watch Demonstration :

![Demonstrate](https://github.com/Jaisood08/Video-Surveillance-System/blob/main/Demonstration.mp4?raw=true)

OUTPUT:

![1Demonstrate](https://github.com/Jaisood08/Video-Surveillance-System/blob/main/Actor_map.png)


How to Use:

1. Clone Repositry.
2. Download dataset from link :
  Dataset :- https://www.kaggle.com/rawatjitesh/avengers-face-recognition
3. Put Dataset in same directory as code files and name it -> dataset
4. Run Train.py for training it will make a embedding.pickle file.
5. Run VideoDetect.py you will get a video and map in ending .
6. If you liked my approch then give it a stark and please provide your valuable suggestions .

Also i have added a yml and requirements file so you can use them to easily run this code.
