import yaml 
import torch
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from src.models.efficientnet import create_model

#load config
config_path = SCRIPT_DIR / "configs" / "config.yaml"
with open (config_path, "r") as f:
    cfg = yaml.safe_load(f)

if __name__ == '__main__':
    print("="*50)
    print("🧪 Testing Model Creation...")
    print("="*50)
    
    #check gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n** Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    #create model
    print("\n Creating model...")
    model = create_model(cfg, device=device)
    
    #test forward pass 
    print("\n Testing forward pass ...")
    dummy_input = torch.randn(2,3,224,224).to(device) #batch size 2, 3 channels, 224x224 image
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  forward pass successful ")
    print(f"  input shape: {dummy_input.shape}")
    print(f"  output shape: {output.shape}") # should be (2, 3)
    print(f"  output (logits): {output}")   
    
    print("\n" + "="*50)
    print("✅ Model Creation Test PASSED!")
    print("="*50)         