"""
Inference script - Make predictions on new mammogram images
"""
import yaml
import torch
from pathlib import Path
import sys
from PIL import Image
import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from src.data.preprocessing import get_transforms
from src.models.efficientnet import create_model


def predict_image(model, image_path, transform, device, class_names):
    """
    Make prediction on a single image
    
    Args:
        model: Trained model
        image_path: Path to image file
        transform: Preprocessing transforms
        device: cuda or cpu
        class_names: List of class names
    
    Returns:
        predicted_class, probabilities
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(1).item()
    
    predicted_class = class_names[predicted_idx]
    class_probs = {name: prob for name, prob in zip(class_names, probs[0].cpu().numpy())}
    
    return predicted_class, class_probs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Mammography Image Classification - Inference')
    parser.add_argument('image_path', type=str, help='Path to mammogram image')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to checkpoint (default: best_model.pth)')
    args = parser.parse_args()
    
    # Load config
    config_path = SCRIPT_DIR / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}\n")
    
    # Load transforms (validation transforms - no augmentation)
    transforms = get_transforms(cfg)
    val_transform = transforms['val']
    
    # Create model
    print("🧠 Loading model...")
    model = create_model(cfg, device=device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Path(cfg['training']['checkpoint_dir']) / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"\n❌ Error: Checkpoint not found at {checkpoint_path}")
        print("   Please train the model first or specify correct checkpoint path")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}\n")
    
    class_names = cfg['data']['class_names']
    
    # Make prediction
    print(f"📷 Analyzing image: {args.image_path}")
    predicted_class, class_probs = predict_image(
        model, args.image_path, val_transform, device, class_names
    )
    
    # Display results
    print("\n" + "="*60)
    print("🎯 PREDICTION RESULTS")
    print("="*60)
    print(f"\n📋 Predicted Class: {predicted_class.upper()}")
    print(f"\n📊 Class Probabilities:")
    for class_name, prob in class_probs.items():
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   {class_name:25s} {bar} {prob*100:6.2f}%")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
