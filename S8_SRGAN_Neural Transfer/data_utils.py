from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

# Images are read from directory.    
# 44, 4 -> parameters passed to 'calculate_crop_size' function. Both parameters are configured.
# Example : Image Input Size -> (127, 224)
# crop_size = (44 - 44%4) = 44
# hr_image.size -> (44,44) ->  i.e. crop size we calculated. We are cropping the hr_image based on crop size from input image.
# lr_image.size -> 44 // 4 = 11 -> (11,11)
# Note : Train data prep is different from validation data prep. In train data, hr_img is significantly down-sized. This is to reduce the training time.

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.plain_transform = plain_transform()
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

# hr_image -> Original image we are giving from validation dataset. We calculate crop size & then again modify same 'hr_image' by CenterCrop from hr_image we started with.
# lr_image -> This is reduced version of original image supplied. Dimension will be crop size//upscale factor
# hr_restore_img -> This is merely resizing the lr_image to make it same size as hr_image. This will be the lr image considered for loss calculations

# Example 1
#   Orig Image size -> (224, 150)
#   150, 4 -> parameters passed to 'calculate_crop_size' function i.e. minimum of original image dimension & upscale factor that we set (in this case 4)
#   crop_size = (150 - 150%4) = 148
#   lr_scale -> 148/4 = 37 i.e. lr_image size will (37, 37)
#   hr_scale -> 148, so hr_image size will be (148,148) i.e. crop size we calculated
#   hr_restore_img_size = (148, 148) because this is mere resizing of lr_image

# Example 2 
#   Orig Image size -> (224, 224)
#   224, 4 -> parameters passed to 'calculate_crop_size' function  i.e. minimum of original image dimension & upscale factor that we set (in this case 4)
#   crop_size = (224 - 224%4) = 224
#   lr_scale -> 224/4 = 56 i.e. lr_image size will (56, 56)
#   hr_scale -> 224, so hr_image size will be (224,224) i.e. crop size we calculated
#   hr_restore_img_size = (224, 224) because this is mere resizing of lr_image

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)