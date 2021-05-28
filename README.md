# Deep_Learning_EVA4_Phase2
Github repository that will hold code base and relevant artifacts related to Extensive Vision AI Phase2 project. Below is summary of applications built.
- Prepared training data and bulit "flying object image recognition" that will classify 4 flying objects - 'Flying Birds', 'Large QuadCopters', 'Small QuadCopters' and 'Winged Drones'. Pretrained Mobilenet-V7 model was customized. 15K images that were downloaded from web was used for training. 
- Prepared training data and built "Indian Car Image generation" based on DC GAN. Model will accept a (100,1) latent vector to generate a (128, 128, 3) size Indian car image.
Training data was prepared by downloading 505 publicly available left front-facing Indian car images with white or no background.
- Built "Indian Car Image Reconstruction" based on VAE (Variational Auto Encoder). Encoder-Decoder architecture was used that will accept an input image and provide reconstructed image as output. Same training data that was used for "Indian Car Image generation" was re-employed.
- Built "Image Super Resolution" that can accept a low resolution image and convert it to high resolution output. Model uses generator-discriminator architecture and was built based on "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" paper. Model was trained on 2K web downloaded flying objects.
- Built "Neural Style Transfer" application that can accept an input image & a style image and convert the input image to suit the style provided on style image.
- Built "Image Captioning" based on "Show-Attend-and-Tell" paper. Model uses ResNet-101 or ResNet-18 as encoder and attention based LSTM network as decoder. Model was trained on 'flickr8k' dataset.
- Built HumanPose Detection model based on "Simple Baseline for HPE and Tracking" arxiv paper. ResNet was used as backbone OpenCV was used for connecting the joints predicted by the model. Model will work for both images & video.
- Built facial landmark detection, face alignment and face swap applications using DLib and OpenCV.
