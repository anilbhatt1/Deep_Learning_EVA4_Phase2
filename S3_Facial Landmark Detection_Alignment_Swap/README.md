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

# Deployment of Resnet, Mobilenet-v2 and Face-Alignment to AWS 
________

# [Link to Web page for Restful API call](http://webdocsridevi.s3-website.ap-south-1.amazonaws.com/)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Face Alignment](#face-alignment)
* [Face Swap](#face-swap)
* [Webpage For Restful Api Call And Integrate It With Aws](#webpage)
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
* [Dlib](http://dlib.net/)

<!-- FACE ALIGNMENT-->
## Face Alignment
- In Face Alignment we will accept an input image and make it aligned as if the face is facing the camera
- We are using 5 point landmark model from Dlib
- 68 point landmark model can also be used but will be resource and time consuming, hence settled for 5 point landmark model
- 5 points landmarked will be two points on left eye corners, two points on right eye corners and one point on nose tip
- https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S3_Facial%20Landmark%20Detection_Alignment_Swap/EVA4P2_S3_Facial_Alignment_5_pt_model_V1.ipynb
- ![Face Aligned image](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S3_Facial%20Landmark%20Detection_Alignment_Swap/Images/Face%20Aligned.jpg)

<!-- FACE SWAP -->
## Face Swap
- We are using Dlib and Open-CV 
- 68 point landmark model from Dlib is used for shape prediction of face (Detecting landmarks)
- dlib.get_frontal_face_detector() is used for face detection
- From the shape detected, we will create a [convex hull](https://medium.com/@pascal.sommer.ch/a-gentle-introduction-to-the-convex-hull-problem-62dfcabee90c#:~:text=The%20convex%20hull%20of%20a,convex%20on%20the%20right%20side.)
- From this convex hull we will create mask then find [Delaunay traingulation](https://en.wikipedia.org/wiki/Delaunay_triangulation#:~:text=In%20mathematics%20and%20computational%20geometry,triangle%20in%20DT(P).) for convex hull points
- We will impose mask of first face over second face and perform [seamless clone](https://docs.opencv.org/master/df/da0/group__photo__clone.html)
- https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S3_Facial%20Landmark%20Detection_Alignment_Swap/EVA4P2_S3_Facial_Swap_Modi_Imran_V1_ipynb.ipynb
- ![Face Swap image](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S3_Facial%20Landmark%20Detection_Alignment_Swap/Images/Modi_Imran_Swapped.jpg)
<!-- WEBPAGE -->
## Webpage for Restful API CALL and Integrate it with AWS

1. Create an S3 bucket
2. Upload to S3 bucket - 5-point landmark.dat, js folder, index.html & error.html.
3. Go to Ubuntu
4. Check if existing environment created for S1 & S2 will suffice. If anything additional or downgrading is required, better to create a dedicated environment for S3 alone with these specific requirements.
5. If not required, activate old environment (S1_mobilenet) itself & proceed.
6. Prepare handler.py in such a way that it points to correct buckets & correct model_paths we chose from web.
   - Option Chosen from web : Resnet34 Classifier -> Should pick S3_BUCKET eva4p2-s1-anilbhatt1 -> MODEL_PATH s1_resnet34.pt
   - Option Chosen from web : MobileNet_V2 Classifier -> Should pick S3_BUCKET eva4p2-s2-anilbhatt1 -> MODEL_PATH s2_mobilenetv2.pt
   - Option Chosen from web : Facealignment  -> Should pick S3_BUCKET eva4p2-s3-anilbhatt1 -> MODEL_PATH shape_predictor_5_face_landmarks.dat
7. Below section in handler.py will need modification to achieve point 6
   - Define environment variables if they are not existing
   - S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'sridevi-session1-bucket'
   - MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'session1mobilenet.pt'
8. We will also need to modify below function in handler.py to get back 'Aligned Face' image
    def get_prediction(image_bytes):
      tensor = transform_image(image_bytes = image_bytes)
      return model(tensor).argmax().item()`

9. Deploy lambda function using serverless, get the url for face alignment.

10. Modify upload.js inside js folder inside AWS S3 bucket to include 3 functions as follows:
    - Create 2 more functions similar to function uploadAndClassifyImage()
    - Total 3 functions - one for Resnet, one for Mobilenet, one for facealignment
    - Modify the function names, give corresponding AWS lambda Urls & also modify the button names accordingly.
    - We can take resnet url (assignment1) from S1 API pathway, mobilnet url (assignment 2) from S2 API pathway and face alignment url (assignment) from S3 API pathway we created in AWS.
11. Accordingly modify 'index.html' inside AWS S3 bucket to accommodate 3 bodies corresponding to 3 functions we created in 'upload.js'

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
