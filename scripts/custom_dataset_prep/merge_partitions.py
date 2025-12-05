import os
import sys
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

# ============ CONFIG ============
INPUT_BASE_DIR = "/data/extracted_datasets"
OUTPUT_DIR = "/data/final_dataset"
NUM_PARTITIONS = 10
BATCH_SIZE = 5000  # Small batch size for memory efficiency
SLEEP_AFTER_MERGE = 10  # seconds
SLEEP_AFTER_SHUFFLE = 5  # seconds
# ================================

train_schema = pa.schema([
    ("id", pa.string()),
    ("emb", pa.list_(pa.float32(), 1024)),
    ("_tenant", pa.string()),
    ("account_id", pa.int64()),
    ("workspace_id", pa.int64()),
    ("ticket_id", pa.int64()),
    ("ticket_type", pa.string()),
    ("ticket_status", pa.string()),
    ("catalog_item_ids", pa.list_(pa.int64())),
    ("created_at", pa.string()),
])


def log(msg):
    """Print with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def get_account_dirs():
    """Get all account directories sorted."""
    base_path = Path(INPUT_BASE_DIR)
    account_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("account_")])
    return account_dirs


def extract_account_id(account_dir):
    """Extract account ID from directory name like 'account_316022_dataset'."""
    name = account_dir.name
    # Format: account_{id}_dataset
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return name


def merge_partition(partition_idx, account_dirs, output_path):
    """Merge all account partition files into one output file."""
    log(f"  Starting merge for partition {partition_idx:02d}...")
    
    writer = None
    total_docs = 0
    accounts_processed = 0
    
    for account_dir in account_dirs:
        account_id = extract_account_id(account_dir)
        part_file = account_dir / f"{account_id}_part_{partition_idx:02d}.parquet"
        
        if not part_file.exists():
            log(f"    WARNING: Missing {part_file}")
            continue
        
        # Stream batches from this partition file
        try:
            pf = pq.ParquetFile(part_file)
            file_docs = 0
            
            for batch in pf.iter_batches(batch_size=BATCH_SIZE):
                if writer is None:
                    writer = pq.ParquetWriter(output_path, train_schema)
                
                table = pa.Table.from_batches([batch], schema=train_schema)
                writer.write_table(table)
                file_docs += batch.num_rows
            
            total_docs += file_docs
            accounts_processed += 1
            
            if accounts_processed % 10 == 0:
                log(f"    Processed {accounts_processed} accounts, {total_docs:,} docs so far...")
                
        except Exception as e:
            log(f"    ERROR reading {part_file}: {e}")
            continue
    
    if writer:
        writer.close()
    
    log(f"  Merge complete: {total_docs:,} docs from {accounts_processed} accounts")
    return total_docs


def shuffle_file(file_path):
    """Shuffle rows within a parquet file."""
    log(f"  Starting shuffle for {file_path.name}...")
    
    try:
        # Read entire file
        df = pl.read_parquet(file_path)
        original_count = len(df)
        log(f"    Loaded {original_count:,} rows into memory")
        
        # Shuffle
        df = df.sample(fraction=1.0, shuffle=True, seed=42)
        log(f"    Shuffled rows")
        
        # Write back
        df.write_parquet(file_path)
        log(f"    Written back to {file_path.name}")
        
        # Free memory
        del df
        
        return original_count
    except Exception as e:
        log(f"    ERROR during shuffle: {e}")
        return 0


def main():
    log("=" * 60)
    log("MERGE PARTITIONS SCRIPT")
    log("=" * 60)
    log(f"Input: {INPUT_BASE_DIR}")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Batch size: {BATCH_SIZE}")
    log(f"Sleep after merge: {SLEEP_AFTER_MERGE}s")
    log(f"Sleep after shuffle: {SLEEP_AFTER_SHUFFLE}s")
    
    # Determine which partitions to process
    if len(sys.argv) > 1:
        partition_idx = int(sys.argv[1])
        partitions_to_process = [partition_idx]
        log(f"Processing SINGLE partition: {partition_idx:02d}")
    else:
        partitions_to_process = list(range(NUM_PARTITIONS))
        log(f"Processing ALL {NUM_PARTITIONS} partitions")
    
    log("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all account directories
    account_dirs = get_account_dirs()
    log(f"Found {len(account_dirs)} account directories")
    
    if len(account_dirs) == 0:
        log("ERROR: No account directories found!")
        sys.exit(1)
    
    # Track doc counts
    partition_counts = {}
    
    # Process each partition
    for partition_idx in partitions_to_process:
        log("")
        log("=" * 60)
        log(f"PARTITION {partition_idx:02d}")
        log("=" * 60)
        
        output_path = Path(OUTPUT_DIR) / f"shuffle_train_{partition_idx:02d}.parquet"
        
        # Step 1: Merge
        doc_count = merge_partition(partition_idx, account_dirs, output_path)
        partition_counts[partition_idx] = doc_count
        
        # Step 2: Sleep after merge
        log(f"  Sleeping {SLEEP_AFTER_MERGE}s after merge...")
        time.sleep(SLEEP_AFTER_MERGE)
        
        # Step 3: Shuffle
        shuffle_file(output_path)
        
        # Step 4: Sleep after shuffle (except for last partition)
        if partition_idx < NUM_PARTITIONS - 1:
            log(f"  Sleeping {SLEEP_AFTER_SHUFFLE}s before next partition...")
            time.sleep(SLEEP_AFTER_SHUFFLE)
    
    # Final Summary
    log("")
    log("=" * 60)
    log("DOC COUNTS FOR THIS RUN")
    log("=" * 60)
    
    total_docs = 0
    min_count = float('inf')
    max_count = 0
    
    for idx in partitions_to_process:
        count = partition_counts.get(idx, 0)
        total_docs += count
        min_count = min(min_count, count) if count > 0 else min_count
        max_count = max(max_count, count)
        log(f"  shuffle_train_{idx:02d}.parquet: {count:,} docs")
    
    log("─" * 40)
    log(f"  Total: {total_docs:,} docs")
    
    if len(partitions_to_process) > 1 and min_count > 0 and min_count != float('inf'):
        log(f"  Min: {min_count:,} | Max: {max_count:,}")
        diff_percent = ((max_count - min_count) / min_count) * 100
        log(f"  Difference: {diff_percent:.2f}%")
    
    log("=" * 60)
    log("COMPLETE!")
    log("=" * 60)


if __name__ == "__main__":
    main()

