#!/usr/bin/env python3
"""
Convert extracted JSON queries to parquet with struct format.

The query is stored as a nested struct (not JSON string) for zero parsing overhead
during benchmarking.

Processes all account folders and creates one parquet file per account.
Supports modifying k and size values for generating different query sets.

Usage:
    # Generate k=50 queries
    python queries_to_parquet.py \
        --input /path/to/json_extraction/ \
        --output /path/to/individual_parquets/ \
        --target-k 50

    # Generate k=100 queries
    python queries_to_parquet.py \
        --input /path/to/json_extraction/ \
        --output /path/to/individual_parquets/ \
        --target-k 100
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def modify_k_and_size(query_dict: dict, target_k: int) -> dict:
    """
    Modify the k and size values in a query dict.
    
    Args:
        query_dict: The full OpenSearch query dict
        target_k: The new k and size value
    
    Returns:
        Modified query dict (original is not mutated)
    """
    import copy
    modified = copy.deepcopy(query_dict)
    
    # Modify size at top level
    if "size" in modified:
        modified["size"] = target_k
    
    # Modify k inside query.knn.<field_name>.k
    if "query" in modified and "knn" in modified["query"]:
        knn = modified["query"]["knn"]
        for field_name, field_data in knn.items():
            if isinstance(field_data, dict) and "k" in field_data:
                field_data["k"] = target_k
    
    return modified


def process_account(account_dir: Path, output_dir: Path, target_k: int) -> int:
    """
    Process all queries for a single account and create one parquet file.
    
    Args:
        account_dir: Directory containing query_*.json files for one account
        output_dir: Directory to write the parquet file
        target_k: The k and size value to set
    
    Returns:
        Number of queries processed
    """
    json_files = sorted(account_dir.glob("query_*.json"))
    if not json_files:
        return 0
    
    rows = []
    for json_file in json_files:
        # Extract query ID from filename: query_<uuid>.json -> <uuid>
        query_id = json_file.stem.replace("query_", "")
        
        # Read JSON query
        with open(json_file, 'r') as f:
            query_dict = json.load(f)
        
        # Modify k and size
        modified_query = modify_k_and_size(query_dict, target_k)
        
        rows.append({
            "id": query_id,
            "query": modified_query
        })
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert all rows to PyArrow table
    # PyArrow automatically handles nested dicts as structs
    ids = [r["id"] for r in rows]
    queries = [r["query"] for r in rows]
    
    table = pa.Table.from_pydict({
        "id": ids,
        "query": queries
    })
    
    # Write to parquet
    output_file = output_dir / "queries.parquet"
    pq.write_table(table, output_file)
    
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON queries to parquet with struct format"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to json_extraction/ directory containing account_id=* folders"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to individual_parquets/ directory"
    )
    parser.add_argument(
        "--target-k",
        type=int,
        required=True,
        help="The k and size value to set in all queries (e.g., 50 or 100)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_base = Path(args.output)
    target_k = args.target_k
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return 1
    
    # Find all account directories
    account_dirs = sorted(input_path.glob("account_id=*"))
    if not account_dirs:
        print(f"Error: No account_id=* directories found in {input_path}")
        return 1
    
    print(f"Found {len(account_dirs)} accounts")
    print(f"Target k={target_k}, size={target_k}")
    print(f"Output: {output_base}/k{target_k}/")
    print()
    
    # Output goes to k{target_k}/ subdirectory
    output_k_dir = output_base / f"k{target_k}"
    
    total_queries = 0
    for account_dir in account_dirs:
        account_name = account_dir.name  # e.g., "account_id=564706"
        account_output = output_k_dir / account_name
        
        count = process_account(account_dir, account_output, target_k)
        total_queries += count
        
        print(f"  {account_name}: {count} queries")
    
    print(f"\n=== Summary ===")
    print(f"Total accounts: {len(account_dirs)}")
    print(f"Total queries: {total_queries}")
    print(f"Output directory: {output_k_dir}")
    
    # Show sample schema
    sample_file = next(output_k_dir.rglob("queries.parquet"), None)
    if sample_file:
        print(f"\nSample parquet schema:")
        table = pq.read_table(sample_file)
        print(table.schema)
    
    return 0


if __name__ == "__main__":
    exit(main())
