from opensearchpy import OpenSearch
from opensearchpy.helpers import scan
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys

# ============ CONFIG ============
OPENSEARCH_HOST = "ip-10-100-20-120.ec2.internal"
OPENSEARCH_PORT = 9200
INDEX_NAME = "dataset_prep_similar_tickets_v2"
OUTPUT_BASE_DIR = "/data/extracted_datasets"
BATCH_SIZE = 500           # OpenSearch scroll batch size
MIN_WRITE_BATCH = 1000     # Minimum docs per parquet write
MAX_WRITE_BATCH = 10000    # Maximum docs per parquet write
NUM_PARTITIONS = 10        # Split into 10 partition files
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


def docs_to_table(docs):
    """Convert list of docs to PyArrow table."""
    return pa.Table.from_pydict({
        "id": [d["id"] for d in docs],
        "emb": [d["emb"] for d in docs],
        "_tenant": [d["_tenant"] for d in docs],
        "account_id": [d["account_id"] for d in docs],
        "workspace_id": [d["workspace_id"] for d in docs],
        "ticket_id": [d["ticket_id"] for d in docs],
        "ticket_type": [d["ticket_type"] for d in docs],
        "ticket_status": [d["ticket_status"] for d in docs],
        "catalog_item_ids": [d["catalog_item_ids"] for d in docs],
        "created_at": [d["created_at"] for d in docs],
    }, schema=train_schema)


def calculate_write_batch_size(total_docs):
    """Calculate optimal write batch size to distribute docs across all partitions."""
    # We want at least NUM_PARTITIONS batches to fill all partitions
    # ideal_batch = total_docs / NUM_PARTITIONS would give 1 batch per partition
    # We want multiple batches per partition for better distribution, so divide by 10-20
    ideal_batch_size = total_docs // (NUM_PARTITIONS * 2)  # ~2 batches per partition minimum
    
    # Clamp between min and max
    return max(MIN_WRITE_BATCH, min(ideal_batch_size, MAX_WRITE_BATCH))


def extract_account(client, account_id):
    """Extract and save parquet partitions for a single account."""
    output_dir = f"{OUTPUT_BASE_DIR}/account_{account_id}_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Extracting docs for account {account_id}...")
    print(f"{'='*50}")
    
    # Pre-count docs to calculate optimal batch size
    query = {"query": {"term": {"account_id": account_id}}}
    count_result = client.count(index=INDEX_NAME, body=query)
    total_docs_expected = count_result["count"]
    
    write_batch_size = calculate_write_batch_size(total_docs_expected)
    print(f"  Total docs: {total_docs_expected:,}, Write batch size: {write_batch_size:,}")
    
    # Initialize writers and counters for each partition
    writers = [None] * NUM_PARTITIONS
    partition_counts = [0] * NUM_PARTITIONS
    partition_paths = [f"{output_dir}/{account_id}_part_{i:02d}.parquet" for i in range(NUM_PARTITIONS)]
    
    docs = []
    total_count = 0
    current_partition = 0
    
    for hit in scan(client, index=INDEX_NAME, query=query, size=BATCH_SIZE, scroll="5m"):
        src = hit["_source"]
        docs.append({
            "id": hit["_id"],
            "emb": src["embedding"],
            "_tenant": src.get("_tenant"),
            "account_id": int(src["account_id"]) if src.get("account_id") else None,
            "workspace_id": int(src["workspace_id"]) if src.get("workspace_id") else None,
            "ticket_id": int(src["ticket_id"]) if src.get("ticket_id") else None,
            "ticket_type": src.get("ticket_type"),
            "ticket_status": src.get("ticket_status"),
            "catalog_item_ids": src.get("catalog_item_ids", []),
            "created_at": src.get("created_at"),
        })
        
        # Write batch to current partition when we hit write_batch_size
        if len(docs) >= write_batch_size:
            table = docs_to_table(docs)
            
            if writers[current_partition] is None:
                writers[current_partition] = pq.ParquetWriter(partition_paths[current_partition], train_schema)
            
            writers[current_partition].write_table(table)
            partition_counts[current_partition] += len(docs)
            total_count += len(docs)
            
            print(f"  Written {total_count:,} docs (partition {current_partition:02d})...")
            
            # Round-robin to next partition
            current_partition = (current_partition + 1) % NUM_PARTITIONS
            docs = []  # Clear memory!
    
    # Write remaining docs to current partition
    if docs:
        table = docs_to_table(docs)
        
        if writers[current_partition] is None:
            writers[current_partition] = pq.ParquetWriter(partition_paths[current_partition], train_schema)
        
        writers[current_partition].write_table(table)
        partition_counts[current_partition] += len(docs)
        total_count += len(docs)
    
    # Close all writers
    for i, writer in enumerate(writers):
        if writer:
            writer.close()
    
    # Print partition summary
    print(f"\n  Partition breakdown for account {account_id}:")
    for i, count in enumerate(partition_counts):
        if count > 0:
            print(f"    part_{i:02d}: {count:,} docs")
    
    print(f"✓ Account {account_id}: {total_count:,} docs saved to {output_dir}/")
    return total_count, partition_counts


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_write_parquet.py <account_id1> [account_id2] [account_id3] ...")
        print("Example: python generate_write_parquet.py 131277 316022 564706")
        print(f"\nMax 10 accounts per run. Each account split into {NUM_PARTITIONS} partitions.")
        print(f"Write batch size auto-adjusts based on account size ({MIN_WRITE_BATCH:,}-{MAX_WRITE_BATCH:,} docs).")
        sys.exit(1)
    
    account_ids = sys.argv[1:]
    
    if len(account_ids) > 10:
        print("Warning: Max 10 accounts allowed. Using first 10.")
        account_ids = account_ids[:10]
    
    print(f"Will process {len(account_ids)} account(s): {', '.join(account_ids)}")
    print(f"Each account will be split into {NUM_PARTITIONS} partitions.")
    
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        use_ssl=False,
    )
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    results = {}
    for account_id in account_ids:
        total, partitions = extract_account(client, account_id)
        results[account_id] = {"total": total, "partitions": partitions}
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    grand_total = 0
    for acc_id, data in results.items():
        non_empty = sum(1 for p in data['partitions'] if p > 0)
        print(f"  Account {acc_id}: {data['total']:,} docs ({non_empty} partitions with data)")
        grand_total += data['total']
    print(f"  ─────────────────────────────")
    print(f"  Grand Total: {grand_total:,} docs")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
