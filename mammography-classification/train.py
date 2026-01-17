import yaml
import torch
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from src.data.preprocessing import get_transforms
from src.data.dataset import create_dataloaders
from src.models.efficientnet import create_model
from src.training.trainer import Trainer


def main():
    #load config
    config_path = SCRIPT_DIR / "configs" / "config.yaml"
    with open (config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    #setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print (f"\n** Using device: {device}")
    if device == 'cuda':
        print (f"   GPU: {torch.cuda.get_device_name(0)}")
    
    #create data loaders
    print("\n loading data...")
    transforms = get_transforms(cfg)
    loaders, sizes, class_names = create_dataloaders(cfg, transforms)   
    print(f"   Train: {sizes['train']} | Val: {sizes['val']} | Test: {sizes['test']}")
    print(f"   Classes: {class_names}\n")        
    
    #create model
    print(" creating model...")
    model = create_model(cfg, device=device)
    print()
    
    #create trainer
    trainer = Trainer(
        model = model,
        train_loader = loaders['train'],
        val_loader = loaders['val'],
        cfg = cfg,
        device = device
        )
    
    #start training 
    trainer.train(num_epochs= cfg['training']['epochs'])
if __name__ == '__main__':
    main()    