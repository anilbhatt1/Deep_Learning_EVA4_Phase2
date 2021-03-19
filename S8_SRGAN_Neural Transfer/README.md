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

# Image Super Resolution Using SRGANs and Neural Style Transfer
________

# [Link to Web page for Restful API call](To be updated)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [SRGAN Working](#SRGAN-Working)
* [Neural Style Transfer Working](#Neural-Style-Transfer-Working)
* [Data Preparation](#Data-Preparation)
* [SRGAN Architecture](#SRGAN-Architecture)
* [Colab Notebook References](#Colab-Notebook-References)
* [Model Weight References](#Model-Weight-References)
* [License](#license)
* [Group Members](#group-members)
* [Mentor](#mentor)

## Prerequisites

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.5.1](https://pytorch.org/) ** Higher versions will cause storage issues while deploying to AWS Lambda 
* [torchvision 0.6.1](https://pytorch.org/docs/stable/torchvision/index.html) ** Higher versions will cause storage issues while deploying to AWS Lambda
* [Google Colab](https://colab.research.google.com/)
* [Open-CV](https://pypi.org/project/opencv-python/)

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
- SSIM : Structural Similarity
  - SSIM (proposed to be closer to human perception compared to PSNR) measures the structural similarity between images in terms of luminance, contrast, and structures.
  - Refer the image below to understand how image quality varies for different PSNR and SSIM combinations
  
<!-- Neural-Style-Transfer-Working -->
## Neural Style Transfer Working
- To be updated.

<!-- Data Preparation -->
## Data Preparation
- To be updated.

<!-- SRGAN-Architecture -->
## SRGAN Architecture
- To be updated.

<!-- Colab-Notebook-References -->
## Colab Notebook References
- To be updated.

<!-- Model-Weight-References -->
## Model Weight References
- To be updated.

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



