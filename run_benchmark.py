import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd_list, desc):
    print("\n" + "="*70, flush=True)
    print(f"STEP: {desc}", flush=True)
    print(f"COMMAND: {' '.join(cmd_list)}", flush=True)
    print("="*70 + "\n", flush=True)
    
    t0 = time.time()
    res = subprocess.run(cmd_list, cwd="/home/sparsh/Naplam/TrenUAV")
    elapsed = (time.time() - t0) / 60.0
    if res.returncode != 0:
        print(f"ERROR: Step failed with returncode {res.returncode}", flush=True)
        sys.exit(res.returncode)
    print(f"\nCOMPLETED: {desc} in {elapsed:.2f} minutes\n", flush=True)

def main():
    print("=================================================================", flush=True)
    print("     STARTING END-TO-END UAV COTTON DISEASE BENCHMARK PIPELINE   ", flush=True)
    print("=================================================================", flush=True)

    base_dir = Path("/home/sparsh/Naplam/TrenUAV")
    
    # 1. Train MobileNetV3-Small (Transfer Learning)
    run_command([
        sys.executable, "train.py",
        "--model", "mobilenet_v3_small",
        "--pretrained",
        "--epochs", "10",
        "--batch_size", "32",
        "--lr", "3e-4"
    ], "Training MobileNetV3-Small (Transfer Learning - 1.5M params)")

    # 2. Train EfficientNet-B3 (Transfer Learning)
    run_command([
        sys.executable, "train.py",
        "--model", "efficientnet_b3",
        "--pretrained",
        "--epochs", "10",
        "--batch_size", "16",
        "--accum_steps", "2",
        "--lr", "3e-4"
    ], "Training EfficientNet-B3 (Transfer Learning - 12.2M params)")

    # 3. Evaluate MobileNetV3-Small
    mb_ckpt = base_dir / "checkpoints/mobilenet_v3_small_transfer/best_model.pt"
    run_command([
        sys.executable, "evaluate.py",
        "--checkpoint", str(mb_ckpt)
    ], "Evaluating MobileNetV3-Small on Test Set")

    # 4. Evaluate EfficientNet-B3
    eff_ckpt = base_dir / "checkpoints/efficientnet_b3_transfer/best_model.pt"
    run_command([
        sys.executable, "evaluate.py",
        "--checkpoint", str(eff_ckpt)
    ], "Evaluating EfficientNet-B3 on Test Set")

    # 5. Generate Comparison Visualizations & Summary
    run_command([
        sys.executable, "compare_experiments.py"
    ], "Generating Final Comparison Report & Plots")

    print("\n=================================================================", flush=True)
    print("                ALL BENCHMARK EXPERIMENTS COMPLETE!              ", flush=True)
    print("=================================================================", flush=True)

if __name__ == '__main__':
    main()
