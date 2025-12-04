import os
import glob
import polars as pl
import numpy as np
import argparse
import math
from pathlib import Path

def process_data(input_dir, output_dir, num_shards=10, rounds_per_shard=10):
    """
    Reads raw data partitioned by account, performs deterministic Round Robin interleaving,
    and saves as Parquet.
    
    Logic:
    1. Split each account into (num_shards * rounds_per_shard) micro-chunks.
    2. Append chunks in a strict loop: Acct1, Acct2... AcctN.
    3. Result: Perfectly interleaved data.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {input_path} for account partitions...")
    
    all_files = []
    if input_path.is_dir():
        # Support JSON (Logstash), rotated files (.json.1), and Parquet
        # Look for any file inside account folders, ignoring hidden ones
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.startswith("."): continue
                # Accept json, json.gz, parquet, or rotated logstash files (usually end in digits)
                if "json" in file or file.endswith(".parquet") or file[-1].isdigit():
                    all_files.append(Path(root) / file)
    
    if not all_files:
        print("No files found!")
        return

    # 1. Load Data per Account
    print(f"Found {len(all_files)} files. Loading and deduplicating...")
    
    data_map = {} # {account_id: DataFrame}
    files_by_account = {}
    
    # Group files by account (assuming folder structure or just greedy load)
    # Since we need to load EVERYTHING to slice it deterministically, we load per account.
    
    for f in all_files:
        parts = str(f).split(os.sep)
        acc_id = "unknown"
        for p in parts:
            if p.startswith("account_id="):
                acc_id = p.split("=")[1]
                break
        if acc_id not in files_by_account:
            files_by_account[acc_id] = []
        files_by_account[acc_id].append(f)
        
    print(f"Identified {len(files_by_account)} accounts.")
    sorted_account_ids = sorted(files_by_account.keys()) # Sort for deterministic order
    
    # Load into memory (Warning: 10M docs might need ~40-60GB RAM if dense)
    # If RAM is an issue, we'd need a lazy reader, but Polars is efficient.
    
    for acc_id in sorted_account_ids:
        files = files_by_account[acc_id]
        dfs = []
        for f in files:
            try:
                if str(f).endswith(".parquet"):
                    df = pl.read_parquet(f)
                else:
                    df = pl.read_ndjson(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if dfs:
            full_df = pl.concat(dfs)
            # DEDUPLICATION SAFETY
            # Assumes 'ticket_id' or 'id' is the unique key. 
            # VectorDBBench usually expects 'id'.
            id_col = "id" if "id" in full_df.columns else "ticket_id"
            if id_col in full_df.columns:
                full_df = full_df.unique(subset=[id_col], keep="first")
                # Rename to 'id' if needed
                if id_col != "id":
                    full_df = full_df.rename({id_col: "id"})
            
            # Rename embedding to 'emb' if needed
            if "embedding" in full_df.columns:
                full_df = full_df.rename({"embedding": "emb"})
                
            data_map[acc_id] = full_df
            print(f"  Loaded Account {acc_id}: {len(full_df)} docs")

    # 2. Calculate Chunk Sizes
    total_micro_chunks = num_shards * rounds_per_shard # e.g., 100
    chunk_sizes = {}
    
    for acc_id, df in data_map.items():
        total_rows = len(df)
        # Micro-chunk size
        size = math.ceil(total_rows / total_micro_chunks)
        chunk_sizes[acc_id] = size
        
    # 3. Round Robin Generation
    print(f"\nStarting Round Robin generation ({num_shards} files, {rounds_per_shard} rounds/file)...")
    
    created_files = []
    
    for file_idx in range(num_shards):
        print(f"Building File {file_idx}...")
        file_buffer = []
        
        for round_idx in range(rounds_per_shard):
            # Global micro-chunk index
            global_chunk_idx = (file_idx * rounds_per_shard) + round_idx
            
            for acc_id in sorted_account_ids:
                df = data_map[acc_id]
                c_size = chunk_sizes[acc_id]
                
                start = global_chunk_idx * c_size
                # Strict slicing handles bounds automatically (returns empty if out of bounds)
                chunk = df.slice(start, c_size)
                
                if len(chunk) > 0:
                    file_buffer.append(chunk)
        
        if not file_buffer:
            print(f"Warning: File {file_idx} is empty!")
            continue
            
        # Concat and Save
        # NO SHUFFLE here - we want the round-robin order preserved
        final_df = pl.concat(file_buffer)
        
        out_name = f"train_part_{file_idx}.parquet"
        out_full = output_path / out_name
        final_df.write_parquet(out_full)
        created_files.append(out_name)
        print(f"  Saved {out_name} ({len(final_df)} rows)")

    print("\nProcessing complete.")
    print("Add this to your CustomDataset config:")
    print(f'train_file: "{",".join([f.replace(".parquet", "") for f in created_files])}"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw data folder")
    parser.add_argument("--output", required=True, help="Path to output parquet folder")
    parser.add_argument("--shards", type=int, default=10, help="Number of output files")
    args = parser.parse_args()
    
    process_data(args.input, args.output, args.shards)
