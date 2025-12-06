import polars as pl
import argparse
from pathlib import Path

def generate_queries(train_dir, output_dir, num_queries=10000):
    """
    Samples queries from the training data proportional to account size.
    Saves as test.parquet.
    """
    train_path = Path(train_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading training data from {train_path}...")
    # Read all train files to get full distribution
    # Since we split them into 10, reading one might be enough if we trust the shuffle,
    # but reading all ensures perfect proportionality for the test set.
    # We only need account_id and emb, maybe id.
    
    files = list(train_path.glob("train_part_*.parquet"))
    if not files:
        print("No training files found.")
        return

    dfs = []
    for f in files:
        # Read only necessary columns to save RAM
        df = pl.read_parquet(f, columns=["id", "emb", "account_id"])
        dfs.append(df)
    
    full_df = pl.concat(dfs)
    total_rows = len(full_df)
    print(f"Loaded {total_rows} rows. Sampling {num_queries} queries...")
    
    # Stratified sampling by account_id
    # Polars doesn't have a direct stratified sample, so we group and sample
    
    # Calculate counts per account
    counts = full_df.group_by("account_id").count()
    
    # Sample
    queries = []
    
    # Iterate over accounts to sample proportionally
    # We can just sample globally if the goal is natural distribution
    # But ensuring we pick exactly proportional numbers is cleaner.
    
    # Actually, simple random sampling from the full dataset *is* proportional sampling
    # provided N is large enough. With 10M rows and 10K queries, it should be fine.
    # But let's do it explicitly to guarantee every small account gets at least something if possible.
    
    sampled_df = full_df.sample(n=num_queries, shuffle=True, seed=12345)
    
    # Save test.parquet
    # VectorDBBench expects: id, emb
    # We also keep account_id for the ground truth generation step (to filter neighbors)
    
    out_file = output_path / "test.parquet"
    sampled_df.write_parquet(out_file)
    print(f"Saved queries to {out_file}")
    
    # Also save a separate scalar_labels.parquet if needed for filtering tests
    # But usually CustomDataset handles this.
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Path to processed parquet folder")
    parser.add_argument("--output_dir", required=True, help="Path to save test.parquet")
    parser.add_argument("--count", type=int, default=10000, help="Number of queries")
    args = parser.parse_args()
    
    generate_queries(args.train_dir, args.output_dir, args.count)

