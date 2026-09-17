import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

COLOR_FOREST = '#1C4A2B'
COLOR_TERRACOTTA = '#B8503A'
COLOR_AMBER = '#D6822A'
COLOR_SAGE = '#7DA286'
COLOR_SLATE = '#2B5C8F'
COLOR_PURPLE = '#8E44AD'
COLOR_GREY = '#7F8C8D'

MODEL_PALETTE = {
    'custom_base_cnn_scratch': ('Base CNN (Scratch)', COLOR_GREY, '--'),
    'mobilenet_v3_small_scratch': ('MobileNetV3 (Scratch)', COLOR_TERRACOTTA, '--'),
    'resnet18_frozen': ('ResNet-18 (Linear Probe)', COLOR_SLATE, '-'),
    'resnet18_partial': ('ResNet-18 (Partial Transfer)', '#3498DB', '-'),
    'densenet121_staged': ('DenseNet-121 (Staged Transfer)', COLOR_PURPLE, '-'),
    'mobilenet_v3_small_transfer': ('MobileNetV3 (Full Transfer)', COLOR_FOREST, '-'),
    'efficientnet_b3_transfer': ('EfficientNet-B3 (Scaled Transfer)', COLOR_AMBER, '-'),
}

def format_display_label(d):
    run_tag = d.get('run_name') or f"{d['model_name']}_{'transfer' if d['pretrained'] else 'scratch'}"
    if run_tag in MODEL_PALETTE:
        return MODEL_PALETTE[run_tag][0].replace(' (', '\n(')
    mode = "Transfer" if d.get('pretrained', False) else "Scratch"
    return f"{d.get('model_name', 'model')}\n({mode})"

def generate_comparison_report(results_dir: Path, output_file: Path, checkpoints_dir: Path):
    metrics_files = list(results_dir.glob("*_metrics.json"))
    if not metrics_files:
        print(f"No metric JSON files found in {results_dir}")
        return

    data = []
    for mf in metrics_files:
        with open(mf) as f:
            data.append(json.load(f))

    # Sort by test accuracy descending
    data = sorted(data, key=lambda x: x['test_accuracy'], reverse=True)

    print("\n" + "="*95)
    print("                 UAV COTTON DISEASE CLASSIFICATION BENCHMARK")
    print("="*95)
    print(f"{'Model Experiment':<36} | {'Params (M)':<11} | {'Accuracy':<10} | {'Macro F1':<10} | {'Latency':<10} | {'FPS':<6}")
    print("-"*95)
    for d in data:
        name = format_display_label(d).replace('\n', ' ')
        params_m = round(d['total_params'] / 1e6, 2)
        print(f"{name:<36} | {params_m:<11} | {d['test_accuracy']:>8.2f}% | {d['macro_f1']:>8.2f}% | {d['latency_ms']:>7.2f} ms | {d['fps']:>6.1f}")
    print("="*95 + "\n")

    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    names = [format_display_label(d) for d in data]
    accuracies = [d['test_accuracy'] for d in data]
    f1_scores = [d['macro_f1'] for d in data]
    latencies = [d['latency_ms'] for d in data]

    x = np.arange(len(names))
    width = 0.35

    # Plot 1: Accuracy & Macro F1 (Earth & Copper Palette)
    axes[0].bar(x - width/2, accuracies, width, label='Test Accuracy (%)', color=COLOR_FOREST, edgecolor='white', linewidth=1)
    axes[0].bar(x + width/2, f1_scores, width, label='Macro F1 (%)', color=COLOR_SAGE, edgecolor='white', linewidth=1)
    axes[0].set_ylabel('Score (%)', fontsize=12, fontweight='bold', color='#28302C')
    axes[0].set_title('Pathology Classification Performance Comparison', fontsize=13, fontweight='bold', pad=12, color=COLOR_FOREST)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9.5)
    axes[0].set_ylim(0, 108)
    axes[0].legend(loc='lower right', framealpha=0.9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    for i in range(len(names)):
        axes[0].text(i - width/2, accuracies[i] + 1.2, f"{accuracies[i]:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color=COLOR_FOREST)
        axes[0].text(i + width/2, f1_scores[i] + 1.2, f"{f1_scores[i]:.1f}%", ha='center', fontsize=8.5, color='#28302C')

    # Plot 2: Edge Latency
    bar_colors = [COLOR_AMBER if 'MobileNet' in n else (COLOR_TERRACOTTA if 'Efficient' in n else COLOR_SLATE) for n in names]
    axes[1].bar(x, latencies, width=0.45, color=bar_colors, edgecolor='white', linewidth=1)
    axes[1].set_ylabel('Inference Latency (ms / image)', fontsize=12, fontweight='bold', color='#28302C')
    axes[1].set_title('Edge UAV Inference Latency (Lower is Better)', fontsize=13, fontweight='bold', pad=12, color=COLOR_TERRACOTTA)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9.5)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    for i in range(len(names)):
        axes[1].text(i, latencies[i] + 0.4, f"{latencies[i]:.1f}ms\n({data[i]['fps']:.0f} FPS)", ha='center', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    plot_path = results_dir / 'models_comparison.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved comparative visualization plot to: {plot_path}")

    # Generate Training Curves Plot
    plot_all_training_curves(checkpoints_dir, results_dir / 'training_curves.png')

def plot_all_training_curves(checkpoints_dir: Path, output_path: Path):
    history_files = list(checkpoints_dir.glob("*/history.json"))
    if not history_files:
        print("No history.json files found for training curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for hf in history_files:
        with open(hf) as f:
            hist = json.load(f)

        run_name = hist.get('run_name') or hf.parent.name
        label, color, ls = MODEL_PALETTE.get(run_name, (run_name, '#333333', '-'))

        epochs = range(1, len(hist['val_loss']) + 1)
        axes[0].plot(epochs, hist['val_loss'], label=label, color=color, linestyle=ls, linewidth=2, marker='o', markersize=4)
        axes[1].plot(epochs, hist['val_acc'], label=label, color=color, linestyle=ls, linewidth=2, marker='s', markersize=4)

    axes[0].set_title('Validation Loss Trajectory Across Epochs', fontsize=13, fontweight='bold', pad=10, color=COLOR_FOREST)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Balanced Cross-Entropy Loss', fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(fontsize=9, loc='upper right', framealpha=0.9)

    axes[1].set_title('Validation Accuracy (%) Across Epochs', fontsize=13, fontweight='bold', pad=10, color=COLOR_FOREST)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    axes[1].set_ylim(20, 102)
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(fontsize=9, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves plot to: {output_path}")

if __name__ == '__main__':
    res_path = Path('/home/sparsh/Naplam/TrenUAV/results')
    ckpt_path = Path('/home/sparsh/Naplam/TrenUAV/checkpoints')
    generate_comparison_report(res_path, res_path / 'comparison_report.md', ckpt_path)
