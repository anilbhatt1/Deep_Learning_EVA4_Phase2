
- This folder deals with creating an AWS Lambda function, deploy it using serverless via Ubuntu to AWS and then classifying an image using Mobilenet network via Lambda with AWS S3.
- Trigger request for AWS Lambda was given via Insomnia.
- An image is uploaded via Insomnia which will trigger AWS Lambda, execute the mobilenet network, classify the image and return the response to Insomnia.
### Overall Flow
![Overall Flow](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/Flow_1.png)
### Test Flow
![Test Flow](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/Test_Flow.jpg)
### Input Image to Mobilenet uploaded via Insomnia
![Input Image](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/Yellow-Labrador-Retriever.jpg)
### Insomnia Response after running Mobilenet via AWS Lambda. Trigger was provided by uploading labrador image through Insomnia
![Mobilenet Response](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S1_MobileNet_AWS_Lambda_S3_Insomnia/outcome1.JPG)
