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

# Deploying MobileNet V2 on AWS ![image](https://github.com/Gaju27/eva4phase2/blob/master/git_store_house/aws.JPG)
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Configuration](#configuration)
    * [Virtual Box](#virtual-box)
    * [AWS](#aws)
    * [Serverless](#serverless)
* [Prerequisites](#prerequisites)
* [License](#license)
* [Group Members](#group-members)
* [Mentor](#mentor)



<!-- CONFIGURATION -->
## Configuration 
   -  #### Virtual Box 
      Please install [Virtual box](https://www.virtualbox.org/wiki/Downloads)
      
      Login with password: `osboxes.org` and after that create your [own user account](https://vitux.com/a-beginners-guide-to-user-management-on-ubuntu/)
      
      Install below mentioned requirements
        
        1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
        2. Visual Studio Code --> you can find this in ubuntu software centre
        3. [Docker](https://docs.docker.com/engine/install/ubuntu/)
        4. [Node.js](https://www.geeksforgeeks.org/installation-of-node-js-on-linux/)
        6. [Serverless](https://www.serverless.com/)
        
   -  #### AWS
         Create user [IAM](https://docs.aws.amazon.com/rekognition/latest/dg/setting-up.html)
         
         Create [S3 BUCKET](https://docs.aws.amazon.com/quickstarts/latest/s3backup/step-1-create-bucket.html)
         
   -  #### Serverless         
         Setup with [Serverless](https://www.serverless.com/framework/docs/providers/aws/cli-reference/config-credentials/)
         
         1. Serverless will help us to deploy AWS lambda function from ubuntu local to AWS :cloud:
         2. Mainly two files are involved in deploying lambda function - [handler.py](https://github.com/Gaju27/eva4phase2/blob/master/Session1/handler.py) and [Serverless.yml](https://github.com/Gaju27/eva4phase2/blob/master/Session1/serverless.yml)
         
## Prerequisites

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [AWS Account](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc)
* [Serverless](https://www.serverless.com/) 
* [Insomnia](https://insomnia.rest/download/)

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

# Results

-   Input Labrador Dog image :point_down: will be uploaded via Insomnia that will trigger AWS Lambda :arrow_right: Execute the Mobilenet network :arrow_right: Classify the input image and return the response to Insomnia. 
![aws_flow](https://github.com/Gaju27/eva4phase2/blob/master/git_store_house/aws_flow.png)

-   MobileNet_V2 is correctly predicting the [class](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) as 208: 'Labrador retriever'
   
![dog](https://github.com/Gaju27/eva4phase2/blob/master/git_store_house/Yellow-Labrador-Retriever.jpg)

![image](https://github.com/Gaju27/eva4phase2/blob/master/Session1/outcome1.JPG)


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
[license-url]: https://github.com/Gaju27/EVA/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555

