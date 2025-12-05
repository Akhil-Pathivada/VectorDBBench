#!/usr/bin/env python3
"""
Merge per-account parquets into a single file, shuffle, and assign sequential IDs.

This script:
1. Reads all per-account parquet files from individual_parquets/k{target_k}/
2. Merges them into a single list
3. Shuffles the queries (randomizes order across accounts)
4. Replaces UUID ids with sequential integers (1 to N)
5. Writes final k{target_k}_queries.parquet

Usage:
    python merge_and_sequence.py \
        --input /path/to/individual_parquets/ \
        --output /path/to/final_parquets/ \
        --target-k 50

    python merge_and_sequence.py \
        --input /path/to/individual_parquets/ \
        --output /path/to/final_parquets/ \
        --target-k 100
"""

import argparse
import random
from pathlib import Path
from typing import List, Dict, Any

import pyarrow as pa
import pyarrow.parquet as pq


def extract_account_id_from_query(query_dict: dict) -> str:
    """
    Extract account_id from the query filter.
    
    The account_id is typically in:
    query.knn.embedding.filter.bool.filter[0].term.account_id
    """
    try:
        knn_clause = query_dict["query"]["knn"]
        field_name = list(knn_clause.keys())[0]
        filter_clause = knn_clause[field_name].get("filter", {})
        
        if "bool" in filter_clause and "filter" in filter_clause["bool"]:
            for f in filter_clause["bool"]["filter"]:
                if f and "term" in f and f["term"] and "account_id" in f["term"]:
                    return str(f["term"]["account_id"])
    except Exception:
        pass
    return "unknown"


def load_all_queries(input_dir: Path, target_k: int) -> List[Dict[str, Any]]:
    """
    Load all queries from per-account parquet files.
    
    Returns:
        List of dicts with keys: original_id, account_id, query
    """
    k_dir = input_dir / f"k{target_k}"
    if not k_dir.exists():
        raise ValueError(f"Directory not found: {k_dir}")
    
    all_queries = []
    account_dirs = sorted(k_dir.glob("account_id=*"))
    
    print(f"Loading queries from {len(account_dirs)} accounts...")
    
    for account_dir in account_dirs:
        parquet_file = account_dir / "queries.parquet"
        if not parquet_file.exists():
            print(f"  Warning: No parquet in {account_dir}")
            continue
        
        # Read parquet
        table = pq.read_table(parquet_file)
        data = table.to_pydict()
        
        # Extract account_id from folder name
        account_id = account_dir.name.replace("account_id=", "")
        
        for i in range(len(data["id"])):
            all_queries.append({
                "original_id": data["id"][i],
                "account_id": account_id,
                "query": data["query"][i]
            })
    
    return all_queries


def main():
    parser = argparse.ArgumentParser(
        description="Merge, shuffle, and sequence query parquets"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to individual_parquets/ directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to final_parquets/ directory"
    )
    parser.add_argument(
        "--target-k",
        type=int,
        required=True,
        help="The k value (50 or 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42 for reproducibility)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    target_k = args.target_k
    seed = args.seed
    
    print("=" * 70)
    print(f"  MERGE AND SEQUENCE (k={target_k})")
    print("=" * 70)
    print(f"Input: {input_path}/k{target_k}/")
    print(f"Output: {output_path}/k{target_k}_queries.parquet")
    print(f"Random seed: {seed}")
    print()
    
    # Step 1: Load all queries
    all_queries = load_all_queries(input_path, target_k)
    print(f"\nLoaded {len(all_queries)} queries total")
    
    # Step 2: Shuffle
    print(f"\nShuffling with seed={seed}...")
    random.seed(seed)
    random.shuffle(all_queries)
    
    # Step 3: Assign sequential IDs (1 to N)
    print("Assigning sequential IDs (1 to N)...")
    for i, q in enumerate(all_queries, start=1):
        q["id"] = i
    
    # Step 4: Create output
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"k{target_k}_queries.parquet"
    
    # Build lists for PyArrow
    ids = [q["id"] for q in all_queries]
    queries = [q["query"] for q in all_queries]
    
    # Create table and write
    table = pa.Table.from_pydict({
        "id": ids,
        "query": queries
    })
    
    pq.write_table(table, output_file)
    
    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"Total queries: {len(all_queries)}")
    print(f"Output file: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Show sample
    print(f"\nSample rows (first 5):")
    for i in range(min(5, len(all_queries))):
        q = all_queries[i]
        account = extract_account_id_from_query(q["query"])
        print(f"  ID {q['id']:4d}: account={account}")
    
    print(f"\nSample rows (last 5):")
    for i in range(max(0, len(all_queries) - 5), len(all_queries)):
        q = all_queries[i]
        account = extract_account_id_from_query(q["query"])
        print(f"  ID {q['id']:4d}: account={account}")
    
    # Verify schema
    print(f"\nParquet schema:")
    result_table = pq.read_table(output_file)
    print(f"  Columns: {result_table.column_names}")
    print(f"  Rows: {result_table.num_rows}")
    print(f"  ID type: {result_table.schema.field('id').type}")
    
    return 0


if __name__ == "__main__":
    exit(main())

