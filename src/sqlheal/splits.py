"""Fixed data splitting utilities for SQLHeal / BIRD benchmark."""
from __future__ import annotations

import collections
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple


def ensure_fixed_splits(
    dev_json_path: str = "data/bird/dev.json",
    output_dir: str = "data/bird",
    tune_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Ensure fixed train/tune (25%) and evaluation (75%) splits exist.
    If they already exist on disk, returns their paths immediately without re-shuffling.
    If not, splits deterministically using the specified seed and stratified by db_id.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tune_path = out_dir / "split_tune.json"
    eval_path = out_dir / "split_eval.json"
    
    # If both splits already exist, do not recreate
    if tune_path.exists() and eval_path.exists():
        return tune_path, eval_path
        
    dev_path = Path(dev_json_path)
    if not dev_path.exists():
        # Look in alternate known locations
        alt_paths = [
            out_dir / "dev/dev_20240627/dev.json",
            out_dir / "dev.json",
        ]
        for alt in alt_paths:
            if alt.exists():
                dev_path = alt
                break
                
    if not dev_path.exists():
        raise FileNotFoundError(f"Cannot create splits: {dev_json_path} not found.")
        
    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Group by database to stratify
    by_db = collections.defaultdict(list)
    for item in data:
        by_db[item.get("db_id", "")].append(item)
        
    rng = random.Random(seed)
    tune_items, eval_items = [], []
    for db_id, items in sorted(by_db.items()):
        items_copy = list(items)
        rng.shuffle(items_copy)
        k = max(1, round(len(items_copy) * tune_ratio))
        tune_items.extend(items_copy[:k])
        eval_items.extend(items_copy[k:])
        
    rng.shuffle(tune_items)
    rng.shuffle(eval_items)
    
    with open(tune_path, "w", encoding="utf-8") as f:
        json.dump(tune_items, f, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, indent=2)
        
    print(f"Created fixed splits: {len(tune_items)} tune items ({tune_path}), {len(eval_items)} eval items ({eval_path})")
    return tune_path, eval_path
