import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_report(results_dir: Path, output_file: Path):
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

    print("\n" + "="*85)
    print("                 UAV COTTON DISEASE CLASSIFICATION BENCHMARK")
    print("="*85)
    print(f"{'Model Experiment':<32} | {'Params (M)':<11} | {'Accuracy':<10} | {'Macro F1':<10} | {'Latency':<10} | {'FPS':<6}")
    print("-"*85)
    for d in data:
        mode = "Transfer" if d['pretrained'] else "Scratch"
        name = f"{d['model_name']} ({mode})"
        params_m = round(d['total_params'] / 1e6, 2)
        print(f"{name:<32} | {params_m:<11} | {d['test_accuracy']:>8.2f}% | {d['macro_f1']:>8.2f}% | {d['latency_ms']:>7.2f} ms | {d['fps']:>6.1f}")
    print("="*85 + "\n")

    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [f"{d['model_name']}\n({'Transfer' if d['pretrained'] else 'Scratch'})" for d in data]
    accuracies = [d['test_accuracy'] for d in data]
    f1_scores = [d['macro_f1'] for d in data]
    latencies = [d['latency_ms'] for d in data]

    x = np.arange(len(names))
    width = 0.35

    # Plot 1: Accuracy & Macro F1
    axes[0].bar(x - width/2, accuracies, width, label='Test Accuracy (%)', color='#2b5c8f')
    axes[0].bar(x + width/2, f1_scores, width, label='Macro F1 (%)', color='#4682b4')
    axes[0].set_ylabel('Score (%)')
    axes[0].set_title('Classification Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(names)):
        axes[0].text(i - width/2, accuracies[i] + 1.5, f"{accuracies[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        axes[0].text(i + width/2, f1_scores[i] + 1.5, f"{f1_scores[i]:.1f}%", ha='center', fontsize=9)

    # Plot 2: Edge Latency (Inference speed)
    axes[1].bar(x, latencies, width=0.45, color='#e07a5f')
    axes[1].set_ylabel('Latency (ms / image)')
    axes[1].set_title('Edge UAV Inference Latency (Lower is Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(names)):
        axes[1].text(i, latencies[i] + 0.5, f"{latencies[i]:.1f}ms\n({data[i]['fps']:.0f} FPS)", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path = results_dir / 'models_comparison.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved comparative visualization plot to: {plot_path}")

if __name__ == '__main__':
    res_path = Path('/home/sparsh/Naplam/TrenUAV/results')
    generate_comparison_report(res_path, res_path / 'comparison_report.md')
