import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_and_split_dataset(raw_dir: Path, output_dir: Path, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    print("Scanning raw dataset directories...")
    all_image_paths = []
    
    # We scan Training Dataset and Test Dataset
    for split_dir_name in ['Training Dataset', 'Test Dataset']:
        split_p = raw_dir / split_dir_name
        if not split_p.exists():
            continue
        for img_p in split_p.rglob('*.jpg'):
            all_image_paths.append(img_p)
            
    print(f"Total raw image files found: {len(all_image_paths)}")
    
    # Deduplicate by content hash
    print("Hashing images for deduplication and leakage prevention...")
    hash_to_entry = {}
    
    for img_p in all_image_paths:
        h = compute_md5(img_p)
        label = img_p.parent.name
        if label == 'Hail damages':
            label = 'Hail damage'
        
        if h not in hash_to_entry:
            hash_to_entry[h] = {
                'image_hash': h,
                'image_path': str(img_p.resolve()),
                'label': label,
                'filename': img_p.name
            }
            
    records = list(hash_to_entry.values())
    print(f"Total unique images after deduplication: {len(records)}")
    
    # Class distribution
    class_counts = Counter(r['label'] for r in records)
    classes = sorted(class_counts.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    print("\nClean ground-truth class distribution:")
    for cls in classes:
        print(f"  {cls} (class {class_to_idx[cls]}): {class_counts[cls]} samples ({class_counts[cls]/len(records)*100:.1f}%)")
        
    for r in records:
        r['label_idx'] = class_to_idx[r['label']]
        
    df = pd.DataFrame(records)
    
    # Stratified Train/Val/Test Split
    rng = np.random.default_rng(seed)
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for cls in classes:
        cls_df = df[df['label'] == cls].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(cls_df)
        n_train = int(np.round(n * train_ratio))
        n_val = int(np.round(n * val_ratio))
        
        train_dfs.append(cls_df.iloc[:n_train])
        val_dfs.append(cls_df.iloc[n_train:n_train + n_val])
        test_dfs.append(cls_df.iloc[n_train + n_val:])
        
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    metadata = {
        'classes': classes,
        'class_to_idx': class_to_idx,
        'train_count': len(train_df),
        'val_count': len(val_df),
        'test_count': len(test_df),
        'total_unique': len(records)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("\n--- Split Generation Completed Successfully ---")
    print(f"Train split: {len(train_df)} images ({len(train_df)/len(records)*100:.1f}%)")
    print(f"Val split:   {len(val_df)} images ({len(val_df)/len(records)*100:.1f}%)")
    print(f"Test split:  {len(test_df)} images ({len(test_df)/len(records)*100:.1f}%)")
    print(f"Saved manifest files to {output_dir}")

if __name__ == '__main__':
    raw_path = Path('/home/sparsh/Naplam/Bangladesh-UAV-PlantDisease')
    out_path = Path('/home/sparsh/Naplam/TrenUAV/data/splits')
    clean_and_split_dataset(raw_path, out_path)
