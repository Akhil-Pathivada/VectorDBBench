import os
import sys
import time
from pathlib import Path
import polars as pl

# ============ CONFIG ============
INPUT_DIR = "/data/final_dataset"
OUTPUT_DIR = "/data/final_dataset_balanced"
TARGET_DOCS_PER_FILE = 1_000_000
NUM_PARTITIONS = 10
SLEEP_AFTER_FILE = 5  # seconds
# ================================


def log(msg):
    """Print with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def main():
    log("=" * 60)
    log("BALANCE PARTITIONS SCRIPT")
    log("=" * 60)
    log(f"Input: {INPUT_DIR}")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Target docs per file: {TARGET_DOCS_PER_FILE:,}")
    log("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========================================
    # PHASE 1: Collect excess from large files (00-04)
    # ========================================
    log("")
    log("=" * 60)
    log("PHASE 1: Collect excess from partitions 00-04")
    log("=" * 60)
    
    excess_dfs = []
    
    for partition_idx in range(5):  # 00 to 04
        input_file = Path(INPUT_DIR) / f"shuffle_train_{partition_idx:02d}.parquet"
        output_file = Path(OUTPUT_DIR) / f"shuffle_train_{partition_idx:02d}.parquet"
        
        log(f"\nProcessing partition {partition_idx:02d}...")
        
        # Read file
        df = pl.read_parquet(input_file)
        total_docs = len(df)
        log(f"  Read {total_docs:,} docs")
        
        if total_docs > TARGET_DOCS_PER_FILE:
            # Split: keep first 1M, excess goes to collection
            keep_df = df.head(TARGET_DOCS_PER_FILE)
            excess_df = df.tail(total_docs - TARGET_DOCS_PER_FILE)
            
            log(f"  Keeping {len(keep_df):,} docs, moving {len(excess_df):,} to excess")
            
            # Write balanced file
            keep_df.write_parquet(output_file)
            log(f"  Written {len(keep_df):,} docs to {output_file.name}")
            
            # Collect excess
            excess_dfs.append(excess_df)
        else:
            # No excess, just copy
            df.write_parquet(output_file)
            log(f"  No excess, copied {total_docs:,} docs to {output_file.name}")
        
        # Free memory
        del df
        
        log(f"  Sleeping {SLEEP_AFTER_FILE}s...")
        time.sleep(SLEEP_AFTER_FILE)
    
    # Combine all excess
    log("\nCombining excess docs...")
    excess_combined = pl.concat(excess_dfs)
    total_excess = len(excess_combined)
    log(f"Total excess collected: {total_excess:,} docs")
    
    # Free memory from individual excess dfs
    del excess_dfs
    
    # ========================================
    # PHASE 2: Distribute excess to small files (05-09)
    # ========================================
    log("")
    log("=" * 60)
    log("PHASE 2: Distribute excess to partitions 05-09")
    log("=" * 60)
    
    excess_offset = 0
    
    for partition_idx in range(5, NUM_PARTITIONS):  # 05 to 09
        input_file = Path(INPUT_DIR) / f"shuffle_train_{partition_idx:02d}.parquet"
        output_file = Path(OUTPUT_DIR) / f"shuffle_train_{partition_idx:02d}.parquet"
        
        log(f"\nProcessing partition {partition_idx:02d}...")
        
        # Read file
        df = pl.read_parquet(input_file)
        current_docs = len(df)
        log(f"  Read {current_docs:,} docs")
        
        # Calculate how many docs to add
        if partition_idx < 9:
            # Partitions 05-08: fill up to 1M
            docs_needed = TARGET_DOCS_PER_FILE - current_docs
        else:
            # Partition 09: gets all remaining excess
            docs_needed = total_excess - excess_offset
        
        if docs_needed > 0 and excess_offset < total_excess:
            # Get docs from excess
            docs_to_add = excess_combined.slice(excess_offset, docs_needed)
            excess_offset += len(docs_to_add)
            
            log(f"  Adding {len(docs_to_add):,} docs from excess")
            
            # Combine
            combined_df = pl.concat([df, docs_to_add])
            
            # Write
            combined_df.write_parquet(output_file)
            log(f"  Written {len(combined_df):,} docs to {output_file.name}")
            
            del combined_df
            del docs_to_add
        else:
            # No docs to add, just copy
            df.write_parquet(output_file)
            log(f"  No excess to add, copied {current_docs:,} docs to {output_file.name}")
        
        # Free memory
        del df
        
        log(f"  Sleeping {SLEEP_AFTER_FILE}s...")
        time.sleep(SLEEP_AFTER_FILE)
    
    # Free excess memory
    del excess_combined
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    log("")
    log("=" * 60)
    log("FINAL VERIFICATION")
    log("=" * 60)
    
    total_docs = 0
    min_count = float('inf')
    max_count = 0
    
    for partition_idx in range(NUM_PARTITIONS):
        output_file = Path(OUTPUT_DIR) / f"shuffle_train_{partition_idx:02d}.parquet"
        
        # Read just metadata to get row count (memory efficient)
        df = pl.scan_parquet(output_file).select(pl.count()).collect()
        count = df.item()
        
        total_docs += count
        min_count = min(min_count, count)
        max_count = max(max_count, count)
        
        log(f"  shuffle_train_{partition_idx:02d}.parquet: {count:,} docs")
    
    log("─" * 40)
    log(f"  Total: {total_docs:,} docs")
    log(f"  Min: {min_count:,} | Max: {max_count:,}")
    
    if min_count > 0:
        diff_percent = ((max_count - min_count) / min_count) * 100
        log(f"  Difference: {diff_percent:.2f}%")
    
    log("=" * 60)
    log("BALANCING COMPLETE!")
    log("=" * 60)


if __name__ == "__main__":
    main()

