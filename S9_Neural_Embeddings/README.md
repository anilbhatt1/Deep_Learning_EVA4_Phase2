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

# Neural Embeddings
________

# [Link to Web page for Restful API call](To be updated)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Tokenization - How to feed text data to neural networks](#tokenization)
* [Why neural embeddings are preferred](#neuralembeddings)
* [Word2Vec - CBOW and Skipgram](#word2vec)
* [Colab Notebook References](#Colab-Notebook-References)
* [Model Weight references for future training](#model-weights)
* [License](#license)
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

<!-- tokenization -->
## Tokenization - How to feed text data to neural networks
- Neural networks only understand numbers. Hence like images, text also needs to be converted to numbers to feed to neural networks.
- We can call this vectorization i.e. converting text into vectors. 
- There are mainly 2 parts involved in vectorization of text - tokenization and numericalization.
- Tokenization means creating tokens out of a given sentence. Example as below:
  "This is Anil Bhatt !" --> ["This", "is", "Anil", "Bhatt", "!"]
- Numericalization means assigning an index for these numbers based on their position in vocabulary. Example as below:
  ["This", "is", "Anil", "Bhatt", "!"] --> [9876, 5346, 567, 1320, 43]
- From where will we get vocabulary ? Building such a vocabulary will be cumbersome. Hence, We usually use standard word vectors like Glove for this purpose.
  Example : "glove.6b.100d" --> Means Glove trained on 6 Billion words and having 100 dimensions.
- Tokenization is also not straight forward process if we choose to build it. Example : "Stock value crashed by $100B for Apple over nite :( ...(sic)" . We 
  can see that there are several words like 100(can be any number), B(here it denotes Billion but can mean different things based on context), sic (means  
  quoted verbatim, hence cannot put under unknown word category), :( (sad smiley, not some random characters).
- Tokenization of above sentence will not be easy. Hence, for tokenization also, we will use standard tokenizers like Spacy.  
- These tokenized inputs are passed to an embedding layer which will give a vector representation of the tokenized word based on its value in embedding   
  layer. Imagine embedding layer as a look-up table. 
  Text Input to embedding layer shape = [batch size, sentence len], Output from embedding layer shape = [batch size, sentence len, embedding dimension]
- This embedding layer output will then be fed to subsequent DNN layers to predict an output. Output could be next word in sequence, sentiment (+ve or -ve as 
  in IMDB movie review dataset), type of question (HUM, ETY, DESC etc as in TREC dataset) etc.

<!-- neuralembeddings -->
## Why neural embeddings are preferred
- 505 images of Indian cars were selected from web.
- All cars selected were front facing as shown below with most of the images with white or no background.
- This selection was done to make the network train with limited resource available via google colab.
- File creation colab link is as below
 https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/EVA4_P2_S6_File_Creation.ipynb
- Enitre input dataset that was used for zip file creation can be found in below drive location
 https://drive.google.com/drive/folders/1SMv5kS5ZrMBbwTO35272Xyw3oB8oCMlE?usp=sharing
- Input image samples

 ![Input Sample Images](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S6_GAN/Readme_Contents/Input%20Sample%20images.png)

<!-- word2vec -->
## Word2Vec - CBOW and Skipgram
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



