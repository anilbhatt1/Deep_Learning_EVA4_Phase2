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

# Training MobileNet V2 against custom dataset having 4 flying objects as classes and then Deploying on AWS ![image](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/aws.jpg)
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [License](#license)
* [Group Members](#group-members)
* [Mentor](#mentor)
* [Approach](#Approach explanation)
* [Resize Strategy](#resize-strategy)
* [Model Trained](#model-trained)
* [Graphs](#graphs)
* [Misclassified Images](#misclassified-images)
        
## Prerequisites

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [AWS Account](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc)
* [Serverless](https://www.serverless.com/) 
* [Insomnia](https://insomnia.rest/download/)
* [Google Colab](https://colab.research.google.com/)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- GROUP MEMBERS -->
## Group Members
  - [Anilkumar N Bhatt](https://github.com/anilbhatt1) , [Anil_on_LinkedIn](https://www.linkedin.com/in/anilkumar-n-bhatt/)
  - [Gajanana Ganjigatti](https://github.com/gaju27) , [Gaju_on_LinkedIn](https://www.linkedin.com/in/gajanana-ganjigatti/)
  - [Maruthi Srinivas](https://github.com/mmaruthi) , [Maruthi_on_LinkedIn](https://www.linkedin.com/in/maruthi-srinivas-m/)
  - [Sridevi B](https://github.com/sridevibonthu) , [Sridevi_on_LinkedIn](https://www.linkedin.com/in/sridevi-bonthu/)
  - [SMAG TEAM](https://github.com/SMAGEVA4/session1/tree/master/Session1) :performing_arts: team github account

<!-- MENTOR -->
## Mentor

* [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) , [The School of A.I.](https://theschoolof.ai/)

<!-- APPROACH -->
## Approach

#### Part : 1  (Data Preparation)
- Took the images from google drive shared location where images are collected by crowd-souring. Total 19318 images belonging to 4 flying objects were collected.
- These images were located in 4 folders 'Flying Birds', 'Large QuadCopters', 'Small QuadCopters' and 'Winged Drones'. 
- Images from all these folders were brought into a zip file named 'ThumbnailData.zip' and saved to personal gdrive folder. Code base listed below.
- https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S2_Mobilenet_QuadCopters_Lambda/EVA4P2_S2_zip_file_creation.ipynb
#### Part: 2 (Model Training)
- Images from ThumbnailData.zip was extracted and split into training and testing datasets based on 70:30 ratio. 
- Following transforms were applied to these images so that more variety of images will be available during training apart from the original ones.
   1) Rotate
   2) Horizontal Flip
   3) Resize
   4) RGB Shift
   5) Normalize (Used channel mean & std-dev of imagenet dataset)
   6) Cutout ( 1 hole with 25% size of image)
- Then pretrained MobileNet-V2 model is downloaded. 
- Except layers 16, 17, 18 and final classifier layer, all remaining layers were frozen so that they will continue to use the pre-trained weights. 
- As Mobilenet-V2 was pre-trained using imagenet dataset it didn't know about the classes (flying objects) that were used in this use-case. 
- To resolve this, final layers were unfrozen by setting requires_grad as True 
- This way model was given chance to learn these classes during training. 
- Also classifier layer was modified to take 4 classes instead of 1000 imagenet classes. Code base listed below.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S2_Mobilenet_QuadCopters_Lambda/EVA4P2_S2_MobilenetV2_V7.ipynb


<!-- RESIZE STRATEGY -->
## Resize Strategy

<!-- MODEL TRAINED -->
## Model Trained

<!-- GRAPHS -->
## Accuracy, Train Loss, Test Loss vs Epochs graphs for train and test

<!-- MISCLASSIFIED IMAGES -->
## Misclassified Images



# Results

-   Input Labrador Dog image :point_down: will be uploaded via Insomnia that will trigger AWS Lambda :arrow_right: Execute the Mobilenet network :arrow_right: Classify the input image and return the response to Insomnia. 
![aws_flow](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/Flow_1.png)

-   MobileNet_V2 is correctly predicting the [class](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) as 208: 'Labrador retriever'
   
![dog](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/Yellow-Labrador-Retriever.jpg)

![image](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/outcome1.JPG)


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


