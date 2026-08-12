
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
def split_csv(input_csv: Path, output_dir: Path, y_column: str, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    df = pd.read_csv(input_csv)

    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)

    val_rel = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_rel, random_state=random_state, shuffle=True)


    y_test = test[[y_column]]             
    test = test.drop(columns=[y_column])  

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    return {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
        "y_test": output_dir / "y_test.csv"
    }

def parse_args():
    p = argparse.ArgumentParser(description="Split CSV into train/val/test and separate y_test")
    
    default_input = PROJECT_ROOT 
    default_output = PROJECT_ROOT 
    
    p.add_argument("--input", "-i", type=Path, default=default_input, help="Input path")
    p.add_argument("--output", "-o", type=Path, default=default_output, help="Output path")
    p.add_argument("--y_column", default='quality', help='The target col')
    p.add_argument("--test-size", type=float, default=0.15, help="Test size (default 0.15)")
    p.add_argument("--val-size", type=float, default=0.15, help="Val size(default 0.15)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.org")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output)
    
    out_files = split_csv(**vars(args))
    print("Saved:")
    for k, v in out_files.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
