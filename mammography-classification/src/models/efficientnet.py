import torch
import torch.nn as nn
import timm 

class MammographyClassifier(nn.Module):
    """
    EfficientNet-based classifier for mammography imagees
    uses transfer learning from ImageNet weights
    """
    def __init__(self, model_name = 'efficientnet_b0', num_classes=3, pretrained=True,dropout_rate=0.3):
        super().__init__()
        
        #load pre-trained efficientnet model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0 #remove classification head 
            )
        #get the number of features from the backbone
        num_features = self.backbone.num_features
        #custom classif head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x):
        #extract features usiing backbone
        features = self.backbone(x)
        #classify 
        output = self.classifier(features)
        return output
def create_model(cfg, device='cuda'):
    """factory function to create model based on config"""
    model = MammographyClassifier(
        model_name= cfg['model']['name'],
        num_classes= cfg['data']['num_classes'],
        pretrained= cfg['model']['pretrained'],
        dropout_rate= cfg['model']['dropout_rate']
    )        
    model = model.to(device)
    #print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {cfg['model']['name']}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return model