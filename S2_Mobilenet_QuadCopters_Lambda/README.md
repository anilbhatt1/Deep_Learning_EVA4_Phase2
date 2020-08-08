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
* [Approach](#Approach)
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
* [python-resize-image](https://pypi.org/project/python-resize-image/)

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
- Input images that were crowd-sourced and stored in shared location were of various sizes. 
- Inorder to bring uniformity resizing was essential. 
- Hence, images were resized to (3, 224, 224) during pre-processing (part-1) listed above. 
- Size (3,224, 224) was chosen because Mobilenet-V2 was pretrained against this input size. 
- Images were opened using PIL -> Converted to RGB -> Then resized to (224, 224) using resizeimage.resize_thumbnail package. 
- resize_thumbnail from python-resize-image was chosen because it maintains the ratio while trying its best to match the specified size. 
- Since the input size images were of varied size this was a good option as it retains the aspect ratio (width:height) while downsizing/upsizing the images.

<!-- MODEL TRAINED -->
## Model Trained
- Torch version was downgraded to torch==1.5.1+cu92 and torchvision==0.6.1+cu92 because AWS lambda was not able to load model against latest 1.6.0 versions due to space constraints.
- Mobilenet-V2 with 3,504,872 parameters was used for training. 
- Pytorch model has 18 convolution layers. 
- Pattern of Conv2d -> BN -> Relu6 -> Conv2d -> BN -> Relu6 -> Conv2d -> BN was used in each layers. 
- First 15 layers used pre-trained weights while 16th, 17th & 18th layers were trained again with flying object input images chosen for this use-case. 
- Also final linear classifier layer was customized to accept 4 custom classes (flying objects) instead of 1000 image-net classes. 
- Model was trained for 20 epochs and achieved a test accuracy of 82.44%.
- Reduce LR on plateau with initial learning rate of 0.3 against a batch-size of 32 was used.
- Optimizer used was SGD with momentum = 0.9. L2_factor of 0.0001 and L1_factor of 0.0005 were also used inside train_loss function.
- Model was saved in 2 ways:
  1) Using torch.script -> Then torch.save. This version was saved whenever model became stable i.e. LR reduces by a factor of 0.1 or when it reaches 80% of total epochs. This is the GPU version and hence can be loaded again while re-training using Cuda GPU.
  2) Using model.to('cpu') -> model.eval() -> torch.jit.trace -> torch.save. This was saved on final epoch. Model was converted to CPU, traced and then saved. This model was uploaded to AWS S3 bucket and was subsequently invoked via AWS Lambda using Insomnia. Model was converted to cpu because lambda function will work only on cpu saved models

<!-- GRAPHS -->
## Accuracy, Train Loss, Test Loss vs Epochs graphs for train and test
![Graphs](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S2_Mobilenet_QuadCopters_Lambda/Images/Train_Test_Accuracies.png)

<!-- MISCLASSIFIED IMAGES -->
## 10 Misclassified Images of each of the 4 classes

- FB -> Flying Birds
- LQ -> Large Quadcopters
- WD -> Winged Drones
- SQ -> Small Quadcopters

![Misclassified Images](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S2_Mobilenet_QuadCopters_Lambda/Images/Misclassified.jpg)

<!-- CORRECTLY CLASSIFIED IMAGES -->
## 25 Correctly classified Images 

![Correctly classified Images](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S2_Mobilenet_QuadCopters_Lambda/Images/Correctly_Classified.jpg)

# Results

-   Input Large Quadcopter image :point_down: will be uploaded via Insomnia that will trigger AWS Lambda :arrow_right: Execute the Mobilenet network :arrow_right: Classify the input image and return the response to Insomnia. 
![aws_flow](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/Flow_1.png)

-   MobileNet_V2 is correctly predicting the [class](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) as 2: 'Large QuadCopters'
   
![Large Quad](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S2_Mobilenet_QuadCopters_Lambda/Images/Test_LargeQuad.jpg)

![image](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S2_Mobilenet_QuadCopters_Lambda/Images/Insomnia_Screenshot.jpg)


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


