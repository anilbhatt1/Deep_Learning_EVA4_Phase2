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
- Tokenization we referred above is word-by-word tokenization which is the most popular one. We can also use characted-by-characted tokenization. Another 
  method is n-grams. An example of bigram is as follows: "My name is Anil !" --> bigram will be "My name", "name is", "is Anil", "Anil !"
- However, use of convolutions eliminates need for using n-grams. Please refer below code bases to understand how convolutions eliminate need for n-grams.
  https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Faster_Sentiment_Analysis_using_FastText.ipynb
  https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Convolutional_Sentiment_Analysis.ipynb

<!-- neuralembeddings -->
## Why neural embeddings are preferred
- There are multiple ways to associate a vector with word - one is one-hot encoding, another is word embeddings.
- Problem with one-hot encoding that it will return a sparse high-dimensional vector which is expensive. Example : Let us say our vocabulary has 100 words 
  and "Therefore" is one among these words. Let us say a sentence having "Therefore" comes up for vectorization. In this case, token for "Therefore" will be 
  a 100 element vector with index corresponding to "Therefore" only having a value of 1 and rest 99 indexes as zeroes.
- As we can clearly see, this is wasteful. Thus came the idea of neural word embeddings.
- Unlike the word vectors obtained via one-hot encoding, word embeddings are learned from data.
- Below image clearly depicts difference between these two.

![One_hot vs Word_Embedding](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/One_hot%20vs%20Word%20Embedding.jpg)

<!-- word2vec -->
## Word2Vec - CBOW and Skipgram
- Word2Vec is a shallow, two-layer neural network which is trained to reconstruct linguistic contexts of words.
- It comes in two flavors, the Continuous Bag-of-Words (CBOW) model and the Skip-Gram model.
- Algorithmically both are similar as shown below:
![Algorith View](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/CBOW_SkipGram.png)
- Example of CBOW vs Skip-Gram is as below:
![Example](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/CBOW_Example.png)

<!-- Colab Notebook References -->
## Colab Notebook References
- Below ipynb notebooks will help us to get a better understanding on NLP models. Details are as below.
### https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Simple_Sentiment_Analysis.ipynb
- Task : Predict sentiment (+ve 1 or -ve 0) from IMDB movie reviews
- We define a simple embedding layer here. Architecture is embedding -> RNN -> fc -> Prediction
- As it is a binary classification problem, we are using nn.BCEWithLogitsLoss() as loss function
- Model randomly predicts sentiments indicating that it is not a useful model

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



