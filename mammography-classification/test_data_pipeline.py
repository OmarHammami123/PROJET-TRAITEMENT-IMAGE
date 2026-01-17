import yaml
import sys
from pathlib import Path

#add src to path
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from src.data.preprocessing import get_transforms
from src.data.dataset import create_dataloaders

#load config
config_path = SCRIPT_DIR / "configs" / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

if __name__ == '__main__':
    print("="*50)
    print("🧪 Testing Data Pipeline...")
    print("="*50)

    #get transforms
    print("\n📋 Getting transforms...")
    transforms = get_transforms(cfg) 
    print("✅ Transforms created successfully.")

    #create dataloaders
    print("\n📋 Creating dataloaders...")
    loaders, sizes, class_names = create_dataloaders(cfg, transforms)

    print("\n📊 Dataset Summary: ")
    print(f"   Classes: {class_names}")
    print(f"   Number of classes: {len(class_names)}")
    for split, size in sizes.items():
        print(f"   {split.capitalize()}: {size} images")

    #test loading a batch
    print("\n🔍 Testing batch loading from train set...")
    train_loader = loaders['train']
    images, labels = next(iter(train_loader))
    print(f"✅ Batch loaded successfully!")
    print(f"   Batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Labels in batch: {labels.tolist()}")
    print(f"   Image dtype: {images.dtype}")
    print(f"   Image min/max: {images.min().item():.4f}/{images.max().item():.4f}")

    print("\n" + "="*50)
    print("✅ Data Pipeline Test PASSED!")
    print("="*50)
print("="*50)    