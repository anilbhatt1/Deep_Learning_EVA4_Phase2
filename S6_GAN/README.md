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
* [GAN How it works](#GAN How it works)
* [Colab Notebook References](#Colab Notebook References)
* [License](#license)
* [Group Members](#group-members)
* [Mentor](#mentor)

## Prerequisites

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [AWS Account](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc)
* [Serverless](https://www.serverless.com/) 
* [Google Colab](https://colab.research.google.com/)
* [Open-CV](https://pypi.org/project/opencv-python/)
* [Html](https://www.w3schools.com/html/)
* [Jquery](https://jquery.com/)

<!-- GAN How it works-->
## GAN How it works
- In Face Alignment we will accept an input image and make it aligned as if the face is facing the camera
- We are using 5 point landmark model from Dlib
- 68 point landmark model can also be used but will be resource and time consuming, hence settled for 5 point landmark model
- 5 points landmarked will be two points on left eye corners, two points on right eye corners and one point on nose tip
- https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S3_Facial%20Landmark%20Detection_Alignment_Swap/EVA4P2_S3_Facial_Alignment_5_pt_model_V1.ipynb

 ![Face Aligned image](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S3_Facial%20Landmark%20Detection_Alignment_Swap/Images/Face%20Aligned.jpg)

<!-- Colab Notebook References -->
## Colab Notebook References
- We are using Dlib and Open-CV 
- 68 point landmark model from Dlib is used for shape prediction of face (Detecting landmarks)
- dlib.get_frontal_face_detector() is used for face detection
- From the shape detected, we will create a [convex hull](https://medium.com/@pascal.sommer.ch/a-gentle-introduction-to-the-convex-hull-problem-62dfcabee90c#:~:text=The%20convex%20hull%20of%20a,convex%20on%20the%20right%20side.)
- From this convex hull we will create mask then find [Delaunay traingulation](https://en.wikipedia.org/wiki/Delaunay_triangulation#:~:text=In%20mathematics%20and%20computational%20geometry,triangle%20in%20DT(P).) for convex hull points
- We will impose mask of first face over second face and perform [seamless clone](https://docs.opencv.org/master/df/da0/group__photo__clone.html)
- https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S3_Facial%20Landmark%20Detection_Alignment_Swap/EVA4P2_S3_Facial_Swap_Modi_Imran_V1_ipynb.ipynb

![Face Swap image](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S3_Facial%20Landmark%20Detection_Alignment_Swap/Images/Modi_Imran_Swapped.jpg)

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

