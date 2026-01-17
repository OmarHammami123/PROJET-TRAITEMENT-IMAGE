import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class CLAHE:
    """contrast limited adaptive histogram equalization (CLAHE)"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    def __call__(self, img):
        """convert PIL imageto CV2 (numpy array) """ 
        img_np = np.array(img)
        
        #check if image is grayscale or color
        if len(img_np.shape)== 3:
            #convert to LAB color space to apply CLAHE on the L channel
            lab= cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b =cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            #grayscale image
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_np = clahe.apply(img_np)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)#convert to 3 channels for model
        return Image.fromarray(img_np)        
def get_transforms(cfg):
    """returns train and val transforms based on config"""
    img_size = tuple(cfg['data']['image_size'])
    mean = cfg['preprocessing']['normalize_mean'] 
    std = cfg['preprocessing']['normalize_std']
    
    #base list of transforms
    transform_list = []
    
    if cfg['preprocessing']['use_clahe']:
        transform_list.append(CLAHE(clip_limit=cfg['preprocessing']['clip_limit']))
    
    transform_list.append(transforms.Resize(img_size))
    
    #trainiing specific (augmentations)
    train_transforms = transforms.Compose(transform_list + [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])    
    #validation/testing transforms(no augmentations)
    val_transforms = transforms.Compose(transform_list + [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return {
        'train': train_transforms,
        'val': val_transforms,
        'test': val_transforms
    }                             