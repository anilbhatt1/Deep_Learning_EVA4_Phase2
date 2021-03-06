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

# Neural Style Transfer & Image Super Resolution Using SRGANs
________

## Link to Heroku Web page for SRGAN and Fast-Neural Style
- https://neural-eyes.herokuapp.com/

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [References](#references)
* [SRGAN Working](#SRGAN-Working)
* [Neural Style Transfer Working](#Neural-Style-Transfer-Working)
* [Data Preparation For SRGANs](#Data-Preparation-For-SRGANs)
* [SRGAN Architecture](#SRGAN-Architecture)
* [Colab Notebook References](#Colab-Notebook-References)
* [Model Weight References](#Model-Weight-References)
* [SRGAN Results](#SRGAN-Results)
* [Neural Style Transfer Results](#Neural-Style-Transfer-Results)
* [Fast Neural style Results](#Fast-Neural-Style-Results)
* [License](#license)
* [Group Members](#group-members)
* [Mentor](#mentor)

## Prerequisites

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.5.0](https://pytorch.org/) ** Higher versions may cause slugspace issues while deploying to Heroku 
* [torchvision 0.6.0](https://pytorch.org/docs/stable/torchvision/index.html) ** Higher versions may cause slugspace  issues while deploying to Heroku
* [Google Colab](https://colab.research.google.com/)
* [Open-CV](https://pypi.org/project/opencv-python/)
* [Pillow](https://pillow.readthedocs.io/en/stable/#)

## References
* [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [Github code reference for SRGAN(leftthomas)](https://github.com/leftthomas/SRGAN)
* [Neural Style Transfer using Pytorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
* [Fast Neural Style Github reference](https://github.com/pytorch/examples/tree/6c8e2bab4d45f2386929c83bb4480c18d2b660fd/fast_neural_style)

<!-- SRGAN-Working -->
## SRGAN Working
- Image Super-Resolution (SR) refers to the process of recovering high-resolution images from low-resolution images. 
- This problem is very challenging and inherently ill-posed since there are always multiple HR images corresponding to a single LR image.
- Real world applications - medical imaging, surveillance etc.
- There are various methods to generate HR images like interpolation. transpose convolution etc.
- Most promising among these are using DNNs with appropriate loss functions. Here we are focusing on such method termed as SRGAN (Super Resolutions Generative Adverserial GAN).
- Loss functions involved in training an SRGAN are as follows:
  - Pixel Loss : measures the pixel-wise difference between two images and mainly include L1 or L2 loss. In some instances, Charbonnier Loss is used instead of L2 loss.
  - Perceptual Loss : 
	  - This loss was introduced in the paper Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGANs).
	  - The authors use a perceptual loss function composed of a content loss and an adversarial loss.
	  - The content loss compares deep features extracted from SR and HR images with a pre-trained VGG network.
	  - Adversarial loss that is used in all GAN-related architectures, helps in fooling the discriminator and generally produces images which have better perceptual quality.
  - Texture Loss:
    - On account that the reconstructed image should have the same style (color, textures, contrast, etc) with the target image, texture loss is introduced in EnhanceNet. 
    - This loss function tries to optimize the Gram matrix of feature outputs inspired by the Style Transfer loss function.  
- As we are contructing HR images from LR images, assessment of HR image quality is very important.
- The image quality assessment includes:
  - subjective methods based on humans' perception
  - objective computational methods (time-saving but often unable to capture the human visual perception):
    - PSNR
    - SSIM
- PSNR : Peak Signal-to-Noise Ratio
  - For Image SR, PSNR is defined via the maximum pixel value (255) and the mean squared error (MSE) between images.
  - Since the PSNR is only related to the pixel-level MSE, only caring about the differences between corresponding pixels instead of visual perception, it often leads to poor performance in representing the reconstruction quality in real scenes, where we're usually more concerned with human perceptions. 
  - An example as below.
  - Hence we need to assess structural similarity too which is closer to human perception.
![Example](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/Readme_Images/Same%20PSNR%20Images.jpg)
- SSIM : Structural Similarity
  - SSIM (proposed to be closer to human perception compared to PSNR) measures the structural similarity between images in terms of luminance, contrast, and structures.
  - Refer the image below to understand how image quality varies for different PSNR and SSIM combinations
 ![PSNR_SSIM](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/Readme_Images/PSNR_SSIM%20Variations.jpeg) 
 
<!-- Neural-Style-Transfer-Working -->
## Neural Style Transfer Working
- Neural-Style, or Neural-Transfer, allows us to take an image and reproduce it with a new artistic style.
- The algorithm takes three images, an input image, a content-image, and a style-image, and changes the input to resemble the content of the content-image and the artistic style of the style-image.
- The principle is simple: we define two distances, one for the content (DC) and one for the style (DS). 
- DC measures how different the content is between two images while DS measures how different the style is between two images. 
- Then, we take a third image, the input, and transform it to minimize both its content-distance with the content-image and its style-distance with the style-image.
- An example as given below
![Neural style example](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/Readme_Images/Neuralstyle.png)

<!-- Data Preparation -->
## Data Preparation For SRGANs
- Same flying objects data that was prepared for S2_Mobilenet was used for training SRGANs.
- Github reference detailing data preparation is as below:
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S2_Mobilenet_QuadCopters_Lambda
- 250 images each from flying birds, small quad, large quad and winged drones were used for training.
- 5 images each from flying birds, small quad, large quad and winged drones that were not part of training was used for evaluation.
- Sample images used are as below.
![Sample Data](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/Readme_Images/Sample%20Data%20SRGAN.jpg)

<!-- SRGAN-Architecture -->
## SRGAN Architecture
- SRGAN architecture created based on "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" paper is as below
![Architecture](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/Readme_Images/SRGAN_Network%20Design.png)
<!-- Colab-Notebook-References -->
## Colab Notebook References
- Colab notebook link for SRGAN that was tried out on flying objects. Original github code built by leftthomas can be found in references section.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/EVA4P2_S8_SRGAN_V2.ipynb
- Colab notebook link for SRGAN evaluation.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/EVA4P2_S8_SRGAN_V3_Eval.ipynb
- Colab notebook link for Neural Style Transfer based on pytorch tutorial.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/EVA4P2_S8_Neural_Transfer_V2.ipynb
- Colab notebook link for Neural Fast Style Transfer that was tried out. Original github code can be found in references section.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/EVA4P2_S8_Fast_Style_Transfer_V1.ipynb

<!-- Model-Weight-References -->
## Model Weight References
- SRGAN weights for Generator(G) and Discriminator(D) after training using flying objects for 999 epochs can be downloaded from below link
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S8_SRGAN_Neural%20Transfer/Model_Weights/SRGAN
- Fast Neural Style transfer weights (both GPU and CPU) after fixing invalid dictionary key can be downloaded from below link. Weights for 4 styles - Mosaic, Udnie, Candy, rain-princess can be found.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S8_SRGAN_Neural%20Transfer/Model_Weights/Fast_Neural

<!-- SRGAN-Results -->
## SRGAN-Results
- Results of better resolution images generated using SRGAN listed as below
![SRGAN Results](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/Readme_Images/SRGAN_Results.png)

<!-- Neural Style Transfer Results -->
## Neural Style Transfer Results
- Results of SRGAN based on pytorch tutorial (check reference) is as below. Same image was used an input & content images.
![Neural Style Results](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/Readme_Images/Style%20Transfer%20Results.png)

<!-- Fast-Neural-Style-Results -->
## Fast Neural style Results
- Fast Neural style was also tried out to generate style images.
- Pre-trained models corresponding to styles - Mosaic, Udnie, Candy and Rain-princess were used for generating images.
- Please check references section for original github code.
- Style images generated based on 4 styles mentioned above corresponding to input image displayed as below.
![Fast Neural Result](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S8_SRGAN_Neural%20Transfer/Readme_Images/Fast_Neural_Results.png)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- GROUP MEMBERS -->
## Group Members
  - [Anilkumar N Bhatt](https://github.com/anilbhatt1) , [Anil_on_LinkedIn](https://www.linkedin.com/in/anilkumar-n-bhatt/)

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



