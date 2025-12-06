import polars as pl
import argparse
from pathlib import Path
import re

def generate_scalar_labels(train_dir, output_dir):
    """
    Extracts metadata (account_id) from training files to create scalar_labels.parquet
    """
    train_path = Path(train_dir)
    output_path = Path(output_dir)
    
    print(f"Scanning {train_path} for train_part_*.parquet...")
    
    # Find all train files
    files = list(train_path.glob("train_part_*.parquet"))
    
    # Sort strictly by the integer index (train_part_0, train_part_1...)
    # to ensure row alignment is preserved.
    def extract_idx(p):
        match = re.search(r'train_part_(\d+)', p.name)
        return int(match.group(1)) if match else 999999
    
    files.sort(key=extract_idx)
    
    if not files:
        print("No training files found!")
        return

    dfs = []
    for f in files:
        print(f"Reading {f}...")
        # Read id and account_id
        # Note: Ensure 'account_id' exists.
        try:
            df = pl.read_parquet(f, columns=["id", "account_id"])
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not dfs:
        print("No data read.")
        return
        
    full_df = pl.concat(dfs)
    
    # Save as scalar_labels.parquet
    out_file = output_path / "scalar_labels.parquet"
    full_df.write_parquet(out_file)
    print(f"Saved {len(full_df)} labels to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Path to processed parquet folder")
    parser.add_argument("--output_dir", required=True, help="Path to save scalar_labels.parquet")
    args = parser.parse_args()
    
    generate_scalar_labels(args.train_dir, args.output_dir)

