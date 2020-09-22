<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Mentor][mentor-shield]][mentor-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# Generative Adversarial Networks (GANs)
________

# [Link to Web page for Restful API call](To be updated)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [GAN Working](#gan-working)
* [Data Preparation](#Data-Preparation)
* [DCGAN DNN Architecture](#DNN-Architecture)
* [Colab Notebook References](#Colab-Notebook-References)
* [Model Weight references for future training](#model-weights)
* [License](#license)
* [Group Members](#group-members)
* [Mentor](#mentor)

## Prerequisites

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.5.1](https://pytorch.org/) ** Higher versions will cause storage issues while deploying to AWS Lambda 
* [torchvision 0.6.1](https://pytorch.org/docs/stable/torchvision/index.html) ** Higher versions will cause storage issues while deploying to AWS Lambda
* [AWS Account](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc)
* [Serverless](https://www.serverless.com/) 
* [Google Colab](https://colab.research.google.com/)
* [Open-CV](https://pypi.org/project/opencv-python/)
* [Html](https://www.w3schools.com/html/)
* [Jquery](https://jquery.com/)

<!-- GAN Working -->
## GAN Working
- Whole idea of GAN is running around 2 deep neural networks - Generator (G) and Discriminator(D)
- Given a random set of latent vectors, Generator(G) will generate images.
- Discriminator (D) will identify whether this image is real or fake. 
- D will be fed with both real and fake images. D loss function will backpropagate based on D's prediction (Real - 1, Fake - 0)
- For each epoch, D will be trained first and G trained next.
- For G also training is done via D. G will generate fake image again and D will predict (Real - 1, Fake - 0). Loss will backpropagate again but this time for G. 
- Below image will help understand the flow Deep Convolutional GAN (DCGAN)

 ![DCGAN Flow](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/Readme_Contents/DCGAN%20Flow%20diagram.jpg)
- Below is an example of DCGAN with MNIST

 ![DCGAN Example with MNIST](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/Readme_Contents/DCGAN%20Flow%20with%20MNIST.jpg)
- This work is dealing with generation of Indian cars using DCGAN (Deep Convolutional GANs)

<!-- Data Preparation -->
## Data Preparation
- 505 images of Indian cars were selected from web.
- All cars selected were front facing as shown below with most of the images with white or no background.
- This selection was done to make the network train with limited resource available via google colab.
- File creation colab link is as below
 https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/EVA4_P2_S6_File_Creation.ipynb
- Enitre input dataset that was used for zip file creation can be found in below drive location
 https://drive.google.com/drive/folders/1SMv5kS5ZrMBbwTO35272Xyw3oB8oCMlE?usp=sharing
- Input image samples

 ![Input Sample Images](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/Readme_Contents/Input%20Sample%20images.png)

<!-- DNN Architecture -->
## DCGAN DNN Architecture
- Two methods were followed as given below.
- First one adopted on Pytorch tutorial https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- Second one adopted based on https://github.com/Yangyangii/GAN-Tutorial/blob/master/CARS/DCGAN.ipynb
- Architecture followed was same in both methods. Difference lies in creating labels while training.
- Refer for receptive field & output size calculations used in architecture - both convolutions and transpose convolutions were used https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/RF%20Calculator_S6_Modelled%20From_Cars_Mine.xlsx
- Network architecture for Generator is as follows:

![Generator Architecture](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/Readme_Contents/Generator%20Network.jpg)

- Network architecture for Discriminator is as follows:

![Discriminator Architecture](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/Readme_Contents/Discriminator%20Network.jpg)

<!-- Colab Notebook References -->
## Colab Notebook References
-	Colab notebook based on Pytorch tutorial approach. Trained for 2000 epochs. Loss function plotted for 500 to 2000 epochs.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/EVA4P2_S6_GAN_V4_Wt_Initialize_Animation_Github.ipynb 
-	Animation of images for 500 to 2000 epochs created based on above training can be referred below:
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/animation_2.mp4
-	Yangyangii approach. Trained for 1400 epochs. Loss function also plotted. Colab notebook reference as below:
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/EVA4P2_S6_GAN_V4_with_weights_labels_using_torch_ones.ipynb
-	Colab notebook reference for Yangyangii approach by loading pretrained model from step above. Trained further for 200 epochs & displayed images of cars generated.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/EVA4P2_S6_GAN_V5_with_weights_labels_using_torch_ones_Images_Displayed.ipynb
- Images generated based on  Yangyangii approach

![Images Generated](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/Readme_Contents/Yangyangii%20Approach_Cars%20Generated.jpg)

<!-- Model weight References -->
## Model Weight references for future training
- Refer below locations to download pretrained weights for future.
- Pytorch tutorial approach
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S6_GAN/Model%20Weights/Pytorch%20Tutorial%20Approach
- Yangyangii approach
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S6_GAN/Model%20Weights/Yangyangii%20Cars%20Example%20Github

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- GROUP MEMBERS -->
## Group Members
  - [Gajanana Ganjigatti](https://github.com/gaju27) , [Gaju_on_LinkedIn](https://www.linkedin.com/in/gajanana-ganjigatti/)
  - [Anilkumar N Bhatt](https://github.com/anilbhatt1) , [Anil_on_LinkedIn](https://www.linkedin.com/in/anilkumar-n-bhatt/)
  - [Sridevi B](https://github.com/sridevibonthu) , [Sridevi_on_LinkedIn](https://www.linkedin.com/in/sridevi-bonthu/)
  - [SMAG TEAM](https://github.com/SMAGEVA4/session1/tree/master/Session1) :performing_arts: team github account

<!-- MENTOR -->
## Mentor

* [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) , [The School of A.I.](https://theschoolof.ai/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[mentor-shield]: https://img.shields.io/badge/Mentor-mentor-yellowgreen
[mentor-url]: https://www.linkedin.com/in/rohanshravan/
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555

