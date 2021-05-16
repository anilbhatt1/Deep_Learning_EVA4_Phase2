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

# Image Captioning 
________

## Link to Heroku Web page for Sentiment Analysis and Multi-class question type
- https://neural-eyes.herokuapp.com/

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Image Captioning - How it works](#Image-Captioning)
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

<!-- Image-Captioning -->
## Image-Captioning
- Image captioning doen here is based on "Show And Tell" paper.
- Please refer https://github.com/anilbhatt1/a-PyTorch-Tutorial-to-Image-Captioning for detailed description.
- Original reference is https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

<!-- Colab-Notebook-References -->
## Colab Notebook References
- Below notebook is based on the original reference given above. Display statements are enabled in decoder network by introducing an index 'px'. 'px' is the number of iterations. Using 'px' display statements are controlled inside decoder network that will help to get an understanding of working of network. Besides, code sections are heavily commented listing out the understanding. Use this notebook to gain an understanding on data is created, how network works & how training/validation are done.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S12_Image_Captioning_and_text_to_images/EVA4P2_S12_ImageCaption_V1.ipynb

<!-- Model-Weight-References-For-Future-Training -->
## Model Weight References For Future Training
- Refer below locations to download pretrained weights for future. 
https://drive.google.com/drive/folders/1Vmb34RHxjtKf19HmfMfwO3sjri_XwY4u?usp=sharings

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



