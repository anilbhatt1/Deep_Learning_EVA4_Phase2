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

# Variational Auto Encoders (VAEs)
________

# [Link to Web page for Restful API call](To be updated)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [VAE Working](#gan-working)
* [Data Preparation](#Data-Preparation)
* [VAE DNN Architecture](#DNN-Architecture)
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

<!-- VAE Working -->
## VAE Working
- Variational Auto-encoders(VAE) are special species of Auto-encoders.
- Hence, let us first understand what Auto-encoders are and how they work. AE typically will have an encoder and a decoder network.
- Encoder network will create latent vector/bottleneck from given input image.
- Decoder network will take the bottleneck and recontruct the image.
- Architecture will look as below. Reconstructed image will be compared against original image via a reconstruction loss as illustrated below. Loss used will be a regression loss like L1Loss, BCE or MSE loss.

 ![AE Working](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/AUTOENCODERS.jpg)
 
 ![Auto Encoder Architecture](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/AE_Architecture.jpg)

- Usecases of auto-encoder includes denoising the image as shown in below image.

 ![Denoising](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/Denoising%20Input.jpg)
 
- However, auto-encoders cant seamlessly interpolate between classes. This is because aut-encoders form cluster of classes which are discontinous as shown below. Latent space they convert their inputs to and where their encoded vectors lie, may not be continuous, or allow easy interpolation.
 
 ![AE_Cluster](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/AE%20Cluster.jpg)
 
- This is where VAEs come into picture.
- Variational Autoencoders (VAEs) have one fundamentally unique property that separates them from vanilla autoencoders, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation.
- It achieves this by doing something that seems rather surprising at first: making its encoder not output an encoding vector of size n, rather, outputting two vectors of size n: a vector of means, μ, and another vector of standard deviations, σ.
- Instead of predicting a point as what vanilla autocoders do, VAE predicts a cloud of point as shown below. 

![VAE_Prediction](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/VAE%20Prediction.jpg)

- Aim of VAE is to predict a smooth continous latent space that doesn't overlap with each other. 

![VAE_Cluster](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/VAE_Cluster.jpg)

- Inorder to achieve this, along with predicting μ and σ, loss function also needs to be modified. If we merely use only reconstruction loss as in AEs, we will end-up a below, where each classes will end-up stacking over another. 

![VAE_Cluster with reconstruct](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/VAE%20Cluster%20with%20reconstuct%20loss%20only.jpg)

-But our aim as stated above is to get a smooth continous latent space that doesn't overlap as shown below. Image on right side shows how MNIST results will look like with VAE with smooth interpolation between classes.

![VAE_Expected_Result](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/vae-result.jpg)

- VAE achieves this by introducing one more loss called KULLBACK-LIEBLER DIVERGENCE (KLD loss). KLD loss is a measure to quantify how similar 2 distributions looks like. As aim of KLD loss is to minimize divergence, it will fight against stacking-up tendency we seen above while maintaining divergence between classes to a minimum.

![KLD](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/KLD.jpg)

- VAE Network will look as below.

![VAE_ntwk](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/VAE_Network.jpg)

- This work is dealing with generation of Indian cars using VAE.

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
## VAE DNN Architecture
- Network architecture for VAE (encoder & decoder were combined in single class) as follows:
![VAE Architecture](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/VAE_DNN_Architecture.jpg)
- For code base of network, please refer below:
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/src/models/VAE_Model.py

<!-- Colab Notebook References -->
## Colab Notebook References
-	Trained for 2000 epochs. Loss function plotted for 500 to 2000 epochs. KLD loss was normalized by (batch_size * channels * img width * img height) to keep it comparable with MSE loss (reconstruction loss). Output images were bad before KLD normalization.
- Latent vector after encoder was not passed via activation function. Instead latent vector was directly passed to reparametrization function to derive sample.
- Colab notebook reference used for training is as below.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/EVA4P2_S7_VAE_V5_KLD_batch_imgsize_no_Sigmoid_for_encoder_module.ipynb 
- Training loss code base can be referenced below
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S7_VAE/src/losses
-	Animation of images for 500 to 2000 epochs created based on above training can be referred below:
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/animation_vae_v1.mp4
- Images generated via VAE vs Original Image
![Images Generated](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S7_VAE/Readme_Content/Generated%20vs%20Original.jpg)

<!-- Model weight References -->
## Model Weight references for future training
- Refer below locations to download pretrained weights for future. Both CPU (jit traced for AWS lambda deployments) and GPU versions(for colab training) are available.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S7_VAE/Model%20Weights

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


