import os
from torchvision import datasets
from torch.utils.data import DataLoader

def create_dataloaders(cfg, transforms):
    """
    create dataloaders for train,val , andd test splits
    """
    data_root= cfg['data']['root_dir']
    batch_size= cfg['data']['batch_size']
    num_workers= cfg['data']['num_workers']
    
    loaders = {}
    sizes = {}
    class_names = []
    for split in ['train','val','test']:
        split_dir = os.path.join(data_root, split)
        
        if not os.path.exists(split_dir):
            print(f"  Warning: {split_dir} does not exist. Skipping {split} dataloader creation.")
            continue
        #use ImageFolder to create dataset-it automatically assigns labels based on subfolder names
        dataset = datasets.ImageFolder(
            root = split_dir,
            transform = transforms[split]
        )
        loaders[split]= DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=(split=='train'), #shuffle only for training
            num_workers = num_workers,
            pin_memory = True #faster data transfer to GPU
        )
        sizes[split]= len(dataset)
        if split == 'train':
            class_names = dataset.classes
    print(f"dataloaders created. classes: {class_names}")
    return loaders, sizes, class_names        