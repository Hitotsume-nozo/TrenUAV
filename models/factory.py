import torch
import torch.nn as nn
import torchvision.models as models

class CustomBaseCNN(nn.Module):
    """
    A standard 4-stage baseline CNN for UAV cotton crop disease classification.
    Trained strictly from scratch without any pretrained weights.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        self.default_resolution = 224
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def build_model(model_name="mobilenet_v3_small", num_classes=5, pretrained=True, freeze_backbone=False):
    """
    Builds a CNN classification model with transfer learning or from scratch.
    
    Args:
        model_name: 'custom_base_cnn', 'mobilenet_v3_small', 'resnet18', 'densenet121', 'efficientnet_b3'
        num_classes: Number of disease classes (5)
        pretrained: If True, loads ImageNet-1K pretrained weights for Transfer Learning.
                    If False, initializes randomly from scratch.
        freeze_backbone: If True:
            - For resnet18: Freezes all backbone features (Linear Probe / Feature Extractor)
            - For densenet121: Freezes early dense blocks (Partial / Staged Fine-Tuning)
            - For mobilenet/efficientnet: Freezes features backbone
    """
    model_name = model_name.lower()
    
    if model_name in ["custom_base_cnn", "base_cnn"]:
        model = CustomBaseCNN(num_classes=num_classes)
        pretrained = False
        freeze_backbone = False
        
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[0].out_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        model.default_resolution = 224
        
    elif model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        model.default_resolution = 300
        
    elif model_name in ["resnet18", "resnet18_partial"]:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        if model_name == "resnet18_partial":
            # Staged Partial Fine-Tuning: freeze early layers (conv1, bn1, layer1, layer2), train layer3, layer4, and fc
            for name, param in model.named_parameters():
                if any(k in name for k in ['conv1', 'bn1', 'layer1', 'layer2']):
                    param.requires_grad = False
        elif freeze_backbone:
            # Linear Probing / Feature Extraction: freeze all conv & bn layers
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
                    
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.default_resolution = 224
        
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        
        if freeze_backbone:
            # Staged Partial Fine-Tuning: freeze early layers, fine-tune deep blocks (denseblock3, transition3, denseblock4, norm5)
            for name, param in model.features.named_parameters():
                if not any(k in name for k in ['denseblock3', 'transition3', 'denseblock4', 'norm5']):
                    param.requires_grad = False
                    
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        model.default_resolution = 224
        
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'custom_base_cnn', 'mobilenet_v3_small', 'resnet18', 'densenet121', or 'efficientnet_b3'.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"--- Model Initialized: {model_name} ---")
    print(f"  Transfer Learning (Pretrained): {pretrained}")
    print(f"  Backbone Frozen / Staged:      {freeze_backbone}")
    print(f"  Total Parameters:              {total_params:,}")
    print(f"  Trainable Parameters:          {trainable_params:,}")
    print(f"  Default Input Resolution:      {model.default_resolution}x{model.default_resolution}")
    
    return model
