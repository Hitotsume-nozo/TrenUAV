import torch
import torch.nn as nn
import torchvision.models as models

def build_model(model_name="mobilenet_v3_small", num_classes=5, pretrained=True, freeze_backbone=False):
    """
    Builds a CNN classification model with transfer learning.
    
    Args:
        model_name: 'mobilenet_v3_small', 'efficientnet_b3', 'resnet18'
        num_classes: Number of disease classes (5)
        pretrained: If True, loads ImageNet-1K pretrained weights for Transfer Learning.
                    If False, initializes randomly from scratch.
        freeze_backbone: If True, freezes all backbone features and only trains the head.
    """
    model_name = model_name.lower()
    
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # In_features for MobileNetV3-Small classifier is 1024
        in_features = model.classifier[0].out_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        model.default_resolution = 224
        
    elif model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # In_features for EfficientNet-B3 is 1536
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        model.default_resolution = 300
        
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
                    
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.default_resolution = 224
        
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'mobilenet_v3_small', 'efficientnet_b3', or 'resnet18'.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"--- Model Initialized: {model_name} ---")
    print(f"  Transfer Learning (Pretrained): {pretrained}")
    print(f"  Backbone Frozen:               {freeze_backbone}")
    print(f"  Total Parameters:              {total_params:,}")
    print(f"  Trainable Parameters:          {trainable_params:,}")
    print(f"  Default Input Resolution:      {model.default_resolution}x{model.default_resolution}")
    
    return model
