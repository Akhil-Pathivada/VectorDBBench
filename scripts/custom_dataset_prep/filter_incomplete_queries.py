#!/usr/bin/env python3
"""
Filter incomplete queries and create new parquets with re-sequenced IDs.

This script:
1. Reads queries parquet and neighbors (GT) parquet
2. Filters out queries where neighbor count < threshold
3. Creates new parquets with sequential IDs (1 to N)
4. Generates ID mapping CSV for validation

Usage:
    # For k=50 (threshold = 10)
    python filter_incomplete_queries.py \
        --queries /path/to/k50_queries.parquet \
        --neighbors /path/to/k50/neighbors.parquet \
        --min-neighbors 10 \
        --output-dir /path/to/filtered_final/k50/

    # For k=100 (threshold = 20)
    python filter_incomplete_queries.py \
        --queries /path/to/k100_queries.parquet \
        --neighbors /path/to/k100/neighbors.parquet \
        --min-neighbors 20 \
        --output-dir /path/to/filtered_final/k100/
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


def remove_nulls(obj):
    """Recursively remove null values from dict/list."""
    if isinstance(obj, dict):
        return {k: remove_nulls(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_nulls(item) for item in obj if item is not None]
    else:
        return obj


def extract_account_id(query_dict: dict) -> str:
    """Extract account_id from the query filter."""
    try:
        cleaned = remove_nulls(query_dict)
        knn_clause = cleaned["query"]["knn"]
        field_name = list(knn_clause.keys())[0]
        filter_clause = knn_clause[field_name].get("filter", {})
        
        if "bool" in filter_clause and "filter" in filter_clause["bool"]:
            for f in filter_clause["bool"]["filter"]:
                if f and "term" in f and "account_id" in f.get("term", {}):
                    return str(f["term"]["account_id"])
    except Exception:
        pass
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Filter incomplete queries and re-sequence IDs"
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="Path to queries parquet (e.g., k50_queries.parquet)"
    )
    parser.add_argument(
        "--neighbors",
        required=True,
        help="Path to neighbors parquet (GT file)"
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        required=True,
        help="Minimum number of neighbors to keep (e.g., 10 for k50, 20 for k100)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for filtered files"
    )
    
    args = parser.parse_args()
    
    queries_path = Path(args.queries)
    neighbors_path = Path(args.neighbors)
    output_dir = Path(args.output_dir)
    min_neighbors = args.min_neighbors
    
    # Validate inputs
    if not queries_path.exists():
        print(f"Error: Queries file not found: {queries_path}")
        return 1
    if not neighbors_path.exists():
        print(f"Error: Neighbors file not found: {neighbors_path}")
        return 1
    
    print("=" * 70)
    print("  FILTER INCOMPLETE QUERIES")
    print("=" * 70)
    print(f"  Queries file:    {queries_path}")
    print(f"  Neighbors file:  {neighbors_path}")
    print(f"  Min neighbors:   {min_neighbors}")
    print(f"  Output dir:      {output_dir}")
    print("=" * 70)
    print()
    
    # Read input files
    print("Reading input files...")
    queries_table = pq.read_table(queries_path)
    neighbors_table = pq.read_table(neighbors_path)
    
    queries_data = queries_table.to_pydict()
    neighbors_data = neighbors_table.to_pydict()
    
    total_queries = len(queries_data["id"])
    total_neighbors = len(neighbors_data["id"])
    
    print(f"  Queries:   {total_queries} rows")
    print(f"  Neighbors: {total_neighbors} rows")
    
    if total_queries != total_neighbors:
        print(f"Error: Row count mismatch! Queries={total_queries}, Neighbors={total_neighbors}")
        return 1
    
    # Filter based on neighbor count
    print(f"\nFiltering queries with >= {min_neighbors} neighbors...")
    
    filtered_queries = []
    filtered_neighbors = []
    id_mapping = []  # (new_id, old_id, account_id, neighbor_count)
    
    removed_count = 0
    new_id = 0
    
    for i in range(total_queries):
        old_id = queries_data["id"][i]
        query = queries_data["query"][i]
        neighbors = neighbors_data["neighbors"][i]
        neighbor_count = len(neighbors) if neighbors else 0
        
        # Check if this query meets the threshold
        if neighbor_count >= min_neighbors:
            new_id += 1
            
            # Extract account_id for mapping
            account_id = extract_account_id(query)
            
            # Add to filtered lists
            filtered_queries.append({
                "id": new_id,
                "query": query
            })
            filtered_neighbors.append({
                "id": new_id,
                "neighbors": neighbors
            })
            id_mapping.append({
                "new_id": new_id,
                "old_id": old_id,
                "account_id": account_id,
                "neighbor_count": neighbor_count
            })
        else:
            removed_count += 1
    
    kept_count = len(filtered_queries)
    
    print(f"  Kept:    {kept_count} queries ({kept_count/total_queries*100:.1f}%)")
    print(f"  Removed: {removed_count} queries ({removed_count/total_queries*100:.1f}%)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write filtered queries parquet
    print(f"\nWriting filtered files...")
    
    queries_output = output_dir / "queries.parquet"
    queries_ids = [q["id"] for q in filtered_queries]
    queries_structs = [q["query"] for q in filtered_queries]
    
    queries_out_table = pa.Table.from_pydict({
        "id": queries_ids,
        "query": queries_structs
    })
    pq.write_table(queries_out_table, queries_output)
    print(f"  Written: {queries_output} ({kept_count} rows)")
    
    # Write filtered neighbors parquet
    neighbors_output = output_dir / "neighbors.parquet"
    neighbors_ids = [n["id"] for n in filtered_neighbors]
    neighbors_lists = [n["neighbors"] for n in filtered_neighbors]
    
    neighbors_out_table = pa.Table.from_pydict({
        "id": neighbors_ids,
        "neighbors": neighbors_lists
    })
    pq.write_table(neighbors_out_table, neighbors_output)
    print(f"  Written: {neighbors_output} ({kept_count} rows)")
    
    # Write ID mapping CSV
    mapping_output = output_dir / "id_mapping.csv"
    with open(mapping_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["new_id", "old_id", "account_id", "neighbor_count"])
        writer.writeheader()
        writer.writerows(id_mapping)
    print(f"  Written: {mapping_output} ({kept_count} rows)")
    
    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Original queries:  {total_queries}")
    print(f"  Threshold:         >= {min_neighbors} neighbors")
    print(f"  Kept:              {kept_count} ({kept_count/total_queries*100:.1f}%)")
    print(f"  Removed:           {removed_count} ({removed_count/total_queries*100:.1f}%)")
    print()
    print(f"  Output files:")
    print(f"    - {queries_output}")
    print(f"    - {neighbors_output}")
    print(f"    - {mapping_output}")
    print()
    print(f"  New ID range: 1 to {kept_count}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

