import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import get_dataloaders
from models.factory import build_model

def compute_detailed_metrics(preds, targets, classes):
    num_classes = len(classes)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for p, t in zip(preds, targets):
        matrix[t, p] += 1
        
    per_class_metrics = {}
    f1_list = []
    
    for i, cls_name in enumerate(classes):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        support = matrix[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)
        
        per_class_metrics[cls_name] = {
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'support': int(support)
        }
        
    overall_acc = np.diag(matrix).sum() / matrix.sum() * 100.0
    macro_f1 = float(np.mean(f1_list)) * 100.0
    
    return overall_acc, macro_f1, per_class_metrics, matrix

def plot_confusion_matrix(matrix, classes, save_path, model_name):
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, cbar=True)
    plt.title(f'Confusion Matrix: {model_name} on UAV Test Set', fontsize=14, pad=15)
    plt.ylabel('True Ground Truth Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")

def benchmark_inference_speed(model, image_size, device, warmup=20, runs=100):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        total_time = time.perf_counter() - t0
        
    avg_latency_ms = (total_time / runs) * 1000.0
    fps = runs / total_time
    return avg_latency_ms, fps

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model on UAV Cotton Disease Test Set")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_model.pt")
    parser.add_argument('--run_name', type=str, default=None, help="Custom run name tag for output files")
    parser.add_argument('--splits_dir', type=str, default='/home/sparsh/Naplam/TrenUAV/data/splits')
    parser.add_argument('--results_dir', type=str, default='/home/sparsh/Naplam/TrenUAV/results')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Evaluating Model on Device: {device} ===")
    
    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    model_name = checkpoint.get('model_name', 'mobilenet_v3_small')
    image_size = checkpoint.get('image_size', 224)
    pretrained = checkpoint.get('pretrained', True)
    freeze_backbone = checkpoint.get('freeze_backbone', False)
    run_name = args.run_name or checkpoint.get('run_name', None)
    
    # Load metadata classes
    with open(Path(args.splits_dir) / 'metadata.json') as f:
        metadata = json.load(f)
    classes = metadata['classes']
    
    # Build Model and load state dict
    model = build_model(model_name=model_name, num_classes=len(classes), pretrained=False, freeze_backbone=freeze_backbone).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Dataloader
    _, _, test_loader = get_dataloaders(
        splits_dir=args.splits_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc, macro_f1, per_class_metrics, conf_mat = compute_detailed_metrics(all_preds, all_targets, classes)
    avg_latency, fps = benchmark_inference_speed(model, image_size, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n========================================================")
    print(f"           TEST RESULTS: {model_name} ({'Transfer Learning' if pretrained else 'From Scratch'})")
    print("========================================================")
    print(f"  Overall Test Accuracy:  {acc:.2f}%")
    print(f"  Macro F1-Score:        {macro_f1:.2f}%")
    print(f"  Inference Latency:     {avg_latency:.2f} ms / image ({fps:.1f} FPS on {device})")
    print(f"  Total Model Params:    {total_params:,}")
    print("--------------------------------------------------------")
    print(f"  {'Class Name':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("--------------------------------------------------------")
    for cls_name, m in per_class_metrics.items():
        print(f"  {cls_name:<30} | {m['precision']:>9.2f}% | {m['recall']:>9.2f}% | {m['f1_score']:>9.2f}% | {m['support']:>7}")
    print("========================================================\n")
    
    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_tag = run_name if run_name else f"{model_name}_{'transfer' if pretrained else 'scratch'}"
    
    output_data = {
        'model_name': model_name,
        'run_name': run_tag,
        'pretrained': pretrained,
        'freeze_backbone': freeze_backbone,
        'total_params': total_params,
        'test_accuracy': round(acc, 2),
        'macro_f1': round(macro_f1, 2),
        'latency_ms': round(avg_latency, 2),
        'fps': round(fps, 1),
        'per_class': per_class_metrics,
        'confusion_matrix': conf_mat.tolist(),
        'classes': classes
    }
    
    with open(results_dir / f"{run_tag}_metrics.json", 'w') as f:
        json.dump(output_data, f, indent=2)
        
    plot_confusion_matrix(conf_mat, classes, results_dir / f"{run_tag}_confusion_matrix.png", f"{model_name} ({run_tag})")

if __name__ == '__main__':
    main()
