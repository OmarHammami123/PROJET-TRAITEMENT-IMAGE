import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path

class MammographySegmentationDataset(Dataset):
    """dataset that pairs mammograms with their segmentation masks"""
    
    def __init__(self, data_dir, csv_path, transform = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        #load csv mapping
        self.data = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        #load image - full mammogram 
        img_path = self.data_dir / row['image_file_path']
        image = Image.open(img_path).convert("L") #convert to grayscale
        
        #load mask
        mask_path = self.data_dir / row['ROI_mask_file_path']
        mask = Image.open(mask_path).convert("L") #grayscale
        
        if self.transform:
            #apply same transform to image and mask
            image = self.transform(image)
            mask = self.transform(mask)
        
        #binarize mask (assuming mask pixels >0 are foreground)
        mask = (mask > 0.5).float()
        
        return image, mask       