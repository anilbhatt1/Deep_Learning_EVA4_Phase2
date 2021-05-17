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

# Image Captioning using DNN
________

## Link to Heroku Web page for Image Captioning
- https://neural-eyes.herokuapp.com/

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Image Captioning Working](#Image-Captioning-Working)
* [Dataset](#Dataset)
* [Colab Notebook References](#Colab-Notebook-References)
* [Model Weight References For Future Training](#Model-Weight-References-For-Future-Training)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.5.0](https://pytorch.org/) ** Higher versions may cause slugspace issues while deploying to Heroku 
* [torchtext 0.6.0](https://pytorch.org/docs/stable/torchvision/index.html) ** Higher versions may cause slugspace  issues while deploying to Heroku
* [Google Colab](https://colab.research.google.com/)

<!-- Image-Captioning-Working-->
## Image Captioning Working
- Image captioning done here is based on "Show And Tell" paper.
- Please refer https://github.com/anilbhatt1/a-PyTorch-Tutorial-to-Image-Captioning for detailed description.
- Original reference is https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

<!-- Dataset -->
## Dataset
- Dataset used for training the model is flickr8k. Original reference listed above is using coco. However, due to space constraints sticked on to flickr8k.
- Datasets can be downloaded from below link.
https://drive.google.com/drive/folders/1b41UGruz0blUbnKGmALyaFtlFgE6MbHe?usp=sharing
- Relevant data file names for training are caption_datasets.zip, Flickr8k_text.zip and Flickr8k_Dataset.zip

<!-- Colab-Notebook-References -->
## Colab Notebook References
- Below notebook is based on the original reference given above. Display statements are enabled in decoder network by introducing an index 'px'. 'px' is the number of iterations. Using 'px' display statements are controlled inside decoder network that will help to get an understanding of working of network. Besides, code sections are heavily commented listing out the understanding. Use this notebook to gain an understanding on data is created, how network works & how training/validation are done.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S12_Image_Captioning_and_text_to_images/EVA4P2_S12_ImageCaption_V1.ipynb
- Below notebook is where training and saving the model weights for future reference is done. Model was originally planned for 50 epochs training. However, stopped at 28th epoch as further improvements were not observed.
https://nbviewer.jupyter.org/github/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S12_Image_Captioning_and_text_to_images/EVA4P2_S12_ImageCaption_V2.ipynb

<!-- Model-Weight-References-For-Future-Training -->
## Model Weight References For Future Training
- Refer below location to download pretrained weights for future puposes. 
https://drive.google.com/drive/folders/1b41UGruz0blUbnKGmALyaFtlFgE6MbHe?usp=sharing

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

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



