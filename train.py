import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

from data.dataset import get_dataloaders
from models.factory import build_model

def compute_macro_f1(preds, targets, num_classes=5):
    """
    Computes macro F1 score without external dependencies.
    """
    f1_scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)), f1_scores

def compute_class_weights(train_csv_path, num_classes=5):
    df = pd.read_csv(train_csv_path)
    counts = df['label_idx'].value_counts().to_dict()
    total = len(df)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        # Standard balanced weighting: total / (num_classes * count)
        w = total / (num_classes * count)
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return weights

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, accum_steps=1):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    optimizer.zero_grad()
    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accum_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accum_steps * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    epoch_acc = (all_preds == all_targets).float().mean().item() * 100.0
    macro_f1, _ = compute_macro_f1(all_preds, all_targets)
    
    return epoch_loss, epoch_acc, macro_f1

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    epoch_acc = (all_preds == all_targets).float().mean().item() * 100.0
    macro_f1, per_class_f1 = compute_macro_f1(all_preds, all_targets)
    
    return epoch_loss, epoch_acc, macro_f1, per_class_f1

def main():
    parser = argparse.ArgumentParser(description="Train CNN with Transfer Learning on UAV Cotton Disease Dataset")
    parser.add_argument('--model', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'efficientnet_b3', 'resnet18'])
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help="Enable Transfer Learning from ImageNet weights")
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help="Train from scratch (no transfer learning)")
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--splits_dir', type=str, default='/home/sparsh/Naplam/TrenUAV/data/splits')
    parser.add_argument('--output_dir', type=str, default='/home/sparsh/Naplam/TrenUAV/checkpoints')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")
    
    # Initialize Model
    model = build_model(
        model_name=args.model,
        num_classes=5,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    image_size = args.image_size if args.image_size else model.default_resolution
    
    # Default batch sizes tuned for 4GB VRAM
    if args.batch_size is None:
        if args.model == 'efficientnet_b3':
            batch_size = 16
            accum_steps = 2
        else:
            batch_size = 32
            accum_steps = 1
    else:
        batch_size = args.batch_size
        accum_steps = args.accum_steps
        
    print(f"Configuration: Resolution={image_size}x{image_size} | Batch Size={batch_size} (Accum={accum_steps}, Effective={batch_size*accum_steps}) | LR={args.lr}")
    
    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        splits_dir=args.splits_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=args.num_workers
    )
    
    # Class-weighted CrossEntropy
    splits_dir = Path(args.splits_dir)
    class_weights = compute_class_weights(splits_dir / 'train.csv', num_classes=5).to(device)
    print("Class Weights for Loss Balancing:", [round(w.item(), 2) for w in class_weights])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    # Setup Output Checkpoint Directory
    mode_str = "transfer" if args.pretrained else "scratch"
    run_name = f"{args.model}_{mode_str}"
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'model': args.model,
        'pretrained': args.pretrained,
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'epoch_times': []
    }
    
    best_val_f1 = 0.0
    best_epoch = 0
    start_total_time = time.time()
    
    print("\n--- Training Progress ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, accum_steps=accum_steps
        )
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        epoch_sec = time.time() - t0
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['epoch_times'].append(epoch_sec)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}", flush=True)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_name': args.model,
                'pretrained': args.pretrained,
                'state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'image_size': image_size
            }, save_dir / 'best_model.pt')
            print(f"  --> Saved new best checkpoint (Val F1: {best_val_f1:.4f})", flush=True)
            
    total_min = (time.time() - start_total_time) / 60.0
    print(f"\nTraining completed in {total_min:.2f} minutes. Best Epoch: {best_epoch} with Val F1: {best_val_f1:.4f}", flush=True)
    
    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"Artifacts saved in: {save_dir}")

if __name__ == '__main__':
    main()
