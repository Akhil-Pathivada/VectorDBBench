#!/usr/bin/env python3
"""
Validate filtered queries by comparing with original files using ID mapping.

This script verifies:
1. Row counts match between filtered files and ID mapping
2. For each mapping entry: query at new_id matches query at old_id in original
3. For each mapping entry: neighbors at new_id match neighbors at old_id in original
4. All neighbor counts in filtered file >= threshold

Usage:
    python validate_filtered_queries.py \
        --original-queries /path/to/k50_queries.parquet \
        --original-neighbors /path/to/k50/neighbors.parquet \
        --filtered-queries /path/to/filtered_final/k50/queries.parquet \
        --filtered-neighbors /path/to/filtered_final/k50/neighbors.parquet \
        --id-mapping /path/to/filtered_final/k50/id_mapping.csv \
        --min-neighbors 10
"""

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq


def remove_nulls(obj):
    """Recursively remove null values from dict/list."""
    if isinstance(obj, dict):
        return {k: remove_nulls(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_nulls(item) for item in obj if item is not None]
    else:
        return obj


def compare_queries(query1: dict, query2: dict) -> bool:
    """Compare two queries after removing nulls."""
    clean1 = remove_nulls(query1)
    clean2 = remove_nulls(query2)
    return clean1 == clean2


def compare_neighbors(neighbors1: list, neighbors2: list) -> bool:
    """Compare two neighbor lists."""
    if neighbors1 is None and neighbors2 is None:
        return True
    if neighbors1 is None or neighbors2 is None:
        return False
    return neighbors1 == neighbors2


def main():
    parser = argparse.ArgumentParser(
        description="Validate filtered queries against original files"
    )
    parser.add_argument(
        "--original-queries",
        required=True,
        help="Path to original queries parquet"
    )
    parser.add_argument(
        "--original-neighbors",
        required=True,
        help="Path to original neighbors parquet"
    )
    parser.add_argument(
        "--filtered-queries",
        required=True,
        help="Path to filtered queries parquet"
    )
    parser.add_argument(
        "--filtered-neighbors",
        required=True,
        help="Path to filtered neighbors parquet"
    )
    parser.add_argument(
        "--id-mapping",
        required=True,
        help="Path to ID mapping CSV"
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        required=True,
        help="Minimum neighbor threshold used for filtering"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of random samples to validate in detail (default: 100)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  VALIDATE FILTERED QUERIES")
    print("=" * 70)
    print(f"  Original queries:   {args.original_queries}")
    print(f"  Original neighbors: {args.original_neighbors}")
    print(f"  Filtered queries:   {args.filtered_queries}")
    print(f"  Filtered neighbors: {args.filtered_neighbors}")
    print(f"  ID mapping:         {args.id_mapping}")
    print(f"  Min neighbors:      {args.min_neighbors}")
    print("=" * 70)
    print()
    
    # Read all files
    print("Reading files...")
    
    orig_queries = pq.read_table(args.original_queries).to_pydict()
    orig_neighbors = pq.read_table(args.original_neighbors).to_pydict()
    filt_queries = pq.read_table(args.filtered_queries).to_pydict()
    filt_neighbors = pq.read_table(args.filtered_neighbors).to_pydict()
    
    # Read ID mapping
    id_mapping = []
    with open(args.id_mapping, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_mapping.append({
                "new_id": int(row["new_id"]),
                "old_id": int(row["old_id"]),
                "account_id": row["account_id"],
                "neighbor_count": int(row["neighbor_count"])
            })
    
    print(f"  Original queries:   {len(orig_queries['id'])} rows")
    print(f"  Original neighbors: {len(orig_neighbors['id'])} rows")
    print(f"  Filtered queries:   {len(filt_queries['id'])} rows")
    print(f"  Filtered neighbors: {len(filt_neighbors['id'])} rows")
    print(f"  ID mapping:         {len(id_mapping)} rows")
    print()
    
    # Build lookup dicts for original data (id -> index)
    orig_query_idx = {orig_queries['id'][i]: i for i in range(len(orig_queries['id']))}
    orig_neighbor_idx = {orig_neighbors['id'][i]: i for i in range(len(orig_neighbors['id']))}
    
    # Validation results
    all_passed = True
    
    # ===== VALIDATION 1: Row counts =====
    print("-" * 70)
    print("  VALIDATION 1: Row counts match")
    print("-" * 70)
    
    filt_query_count = len(filt_queries['id'])
    filt_neighbor_count = len(filt_neighbors['id'])
    mapping_count = len(id_mapping)
    
    if filt_query_count == filt_neighbor_count == mapping_count:
        print(f"  ✅ PASS: All counts match ({filt_query_count} rows)")
    else:
        print(f"  ❌ FAIL: Count mismatch!")
        print(f"     Filtered queries:   {filt_query_count}")
        print(f"     Filtered neighbors: {filt_neighbor_count}")
        print(f"     ID mapping:         {mapping_count}")
        all_passed = False
    print()
    
    # ===== VALIDATION 2: IDs are sequential =====
    print("-" * 70)
    print("  VALIDATION 2: IDs are sequential (1 to N)")
    print("-" * 70)
    
    expected_ids = list(range(1, filt_query_count + 1))
    query_ids_ok = filt_queries['id'] == expected_ids
    neighbor_ids_ok = filt_neighbors['id'] == expected_ids
    mapping_ids_ok = [m['new_id'] for m in id_mapping] == expected_ids
    
    if query_ids_ok and neighbor_ids_ok and mapping_ids_ok:
        print(f"  ✅ PASS: All IDs are sequential 1 to {filt_query_count}")
    else:
        print(f"  ❌ FAIL: IDs not sequential!")
        if not query_ids_ok:
            print(f"     Filtered queries IDs: FAIL")
        if not neighbor_ids_ok:
            print(f"     Filtered neighbors IDs: FAIL")
        if not mapping_ids_ok:
            print(f"     ID mapping new_ids: FAIL")
        all_passed = False
    print()
    
    # ===== VALIDATION 3: Neighbor counts >= threshold =====
    print("-" * 70)
    print(f"  VALIDATION 3: All neighbor counts >= {args.min_neighbors}")
    print("-" * 70)
    
    below_threshold = []
    for i, neighbors in enumerate(filt_neighbors['neighbors']):
        count = len(neighbors) if neighbors else 0
        if count < args.min_neighbors:
            below_threshold.append((filt_neighbors['id'][i], count))
    
    if not below_threshold:
        print(f"  ✅ PASS: All {filt_neighbor_count} queries have >= {args.min_neighbors} neighbors")
    else:
        print(f"  ❌ FAIL: {len(below_threshold)} queries below threshold!")
        for qid, count in below_threshold[:10]:
            print(f"     ID {qid}: {count} neighbors")
        if len(below_threshold) > 10:
            print(f"     ... and {len(below_threshold) - 10} more")
        all_passed = False
    print()
    
    # ===== VALIDATION 4: Query content matches =====
    print("-" * 70)
    print(f"  VALIDATION 4: Query content matches (sampling {min(args.sample_size, filt_query_count)} rows)")
    print("-" * 70)
    
    import random
    sample_indices = random.sample(range(filt_query_count), min(args.sample_size, filt_query_count))
    
    query_mismatches = []
    for i in sample_indices:
        new_id = filt_queries['id'][i]
        mapping_entry = id_mapping[i]
        old_id = mapping_entry['old_id']
        
        # Get queries
        filt_query = filt_queries['query'][i]
        orig_idx = orig_query_idx.get(old_id)
        
        if orig_idx is None:
            query_mismatches.append((new_id, old_id, "old_id not found in original"))
            continue
        
        orig_query = orig_queries['query'][orig_idx]
        
        if not compare_queries(filt_query, orig_query):
            query_mismatches.append((new_id, old_id, "content mismatch"))
    
    if not query_mismatches:
        print(f"  ✅ PASS: All {len(sample_indices)} sampled queries match")
    else:
        print(f"  ❌ FAIL: {len(query_mismatches)} query mismatches!")
        for new_id, old_id, reason in query_mismatches[:10]:
            print(f"     new_id={new_id}, old_id={old_id}: {reason}")
        all_passed = False
    print()
    
    # ===== VALIDATION 5: Neighbor content matches =====
    print("-" * 70)
    print(f"  VALIDATION 5: Neighbor content matches (sampling {min(args.sample_size, filt_neighbor_count)} rows)")
    print("-" * 70)
    
    neighbor_mismatches = []
    for i in sample_indices:
        new_id = filt_neighbors['id'][i]
        mapping_entry = id_mapping[i]
        old_id = mapping_entry['old_id']
        
        # Get neighbors
        filt_neighbor = filt_neighbors['neighbors'][i]
        orig_idx = orig_neighbor_idx.get(old_id)
        
        if orig_idx is None:
            neighbor_mismatches.append((new_id, old_id, "old_id not found in original"))
            continue
        
        orig_neighbor = orig_neighbors['neighbors'][orig_idx]
        
        if not compare_neighbors(filt_neighbor, orig_neighbor):
            neighbor_mismatches.append((new_id, old_id, "content mismatch"))
    
    if not neighbor_mismatches:
        print(f"  ✅ PASS: All {len(sample_indices)} sampled neighbors match")
    else:
        print(f"  ❌ FAIL: {len(neighbor_mismatches)} neighbor mismatches!")
        for new_id, old_id, reason in neighbor_mismatches[:10]:
            print(f"     new_id={new_id}, old_id={old_id}: {reason}")
        all_passed = False
    print()
    
    # ===== VALIDATION 6: Mapping neighbor_count matches actual =====
    print("-" * 70)
    print("  VALIDATION 6: Mapping neighbor_count matches actual count")
    print("-" * 70)
    
    count_mismatches = []
    for i in range(filt_neighbor_count):
        actual_count = len(filt_neighbors['neighbors'][i]) if filt_neighbors['neighbors'][i] else 0
        mapping_count = id_mapping[i]['neighbor_count']
        
        if actual_count != mapping_count:
            count_mismatches.append((filt_neighbors['id'][i], mapping_count, actual_count))
    
    if not count_mismatches:
        print(f"  ✅ PASS: All neighbor counts in mapping match actual")
    else:
        print(f"  ❌ FAIL: {len(count_mismatches)} count mismatches!")
        for qid, expected, actual in count_mismatches[:10]:
            print(f"     ID {qid}: mapping says {expected}, actual is {actual}")
        all_passed = False
    print()
    
    # ===== FINAL RESULT =====
    print("=" * 70)
    if all_passed:
        print("  ✅ ALL VALIDATIONS PASSED!")
    else:
        print("  ❌ SOME VALIDATIONS FAILED - CHECK ABOVE")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

