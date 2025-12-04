#!/usr/bin/env python3
"""
Generate Ground Truth (neighbors.parquet) for custom query dataset.

This script:
1. Reads queries from the final sequenced parquet file (1600 queries)
2. Optionally filters by account_id for testing
3. Converts each approximate KNN query to exact KNN (script_score)
4. Fires queries ONE BY ONE against OpenSearch (with optional sleep between queries)
5. Saves results to neighbors.parquet with SAME integer IDs as query parquet
6. Reports queries that returned fewer than expected results

Usage (Test with specific account):
    python generate_ground_truth.py \
        --input /path/to/sequenced_query_parquet/k50_queries.parquet \
        --output /path/to/ground_truth/k50/ \
        --opensearch-url http://localhost:9200 \
        --index vdb_bench_index \
        --top-k 100 \
        --account-filter 290438

Usage (Production - all queries with sleep):
    python generate_ground_truth.py \
        --input /path/to/sequenced_query_parquet/k50_queries.parquet \
        --output /path/to/ground_truth/k50/ \
        --opensearch-url https://prod-opensearch:9200 \
        --index prod_index \
        --top-k 100 \
        --sleep-ms 100
"""

import argparse
import time
from pathlib import Path
from typing import Optional, List

import pyarrow as pa
import pyarrow.parquet as pq
import requests


def remove_nulls(obj):
    """
    Recursively remove null values from dict/list.
    
    This is needed because parquet stores queries as structs,
    and schema merging adds null values for missing fields.
    OpenSearch fails to parse queries with null values.
    """
    if isinstance(obj, dict):
        return {k: remove_nulls(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_nulls(item) for item in obj if item is not None]
    else:
        return obj


def extract_account_id(query_dict: dict) -> Optional[str]:
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
    return None


def build_exact_knn_query(original_query: dict, top_k: int = 100) -> dict:
    """
    Convert approximate KNN query to exact KNN (script_score) query.
    
    Approximate KNN uses HNSW index (fast but approximate).
    Exact KNN uses script_score (slow but exact results).
    
    Args:
        original_query: The original OpenSearch KNN query (cleaned of nulls)
        top_k: Number of results to return
    
    Returns:
        Exact KNN query using script_score
    """
    # Extract vector and filter from original query
    knn_clause = original_query["query"]["knn"]
    
    # Get the field name (e.g., "embedding")
    field_name = list(knn_clause.keys())[0]
    knn_config = knn_clause[field_name]
    
    vector = knn_config["vector"]
    filter_clause = knn_config.get("filter", None)
    
    # Build filter for script_score query
    if filter_clause and "bool" in filter_clause and "filter" in filter_clause["bool"]:
        filters = filter_clause["bool"]["filter"]
    else:
        filters = []
    
    # Build exact KNN query using script_score
    exact_query = {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": filters
                    }
                },
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": field_name,
                        "query_value": vector,
                        "space_type": "cosinesimil"
                    }
                }
            }
        },
        "size": top_k,
        "_source": False,
        "stored_fields": ["_id"]
    }
    
    return exact_query


def execute_query(opensearch_url: str, index: str, query: dict) -> List[str]:
    """
    Execute a SINGLE query against OpenSearch and return result IDs.
    
    NOTE: We fire queries ONE BY ONE, not in bulk.
    Exact KNN (script_score) is computationally expensive.
    
    Args:
        opensearch_url: OpenSearch endpoint
        index: Index name
        query: Query body
    
    Returns:
        List of document IDs (strings)
    """
    url = f"{opensearch_url}/{index}/_search"
    
    response = requests.post(
        url,
        json=query,
        headers={"Content-Type": "application/json"},
        timeout=120
    )
    
    result = response.json()
    
    if "error" in result:
        raise Exception(f"OpenSearch error: {result['error']}")
    
    # Extract document IDs (keep as strings)
    doc_ids = [hit["_id"] for hit in result["hits"]["hits"]]
    
    return doc_ids


def generate_ground_truth(
    input_file: Path,
    output_dir: Path,
    opensearch_url: str,
    index: str,
    top_k: int = 100,
    account_filter: Optional[str] = None,
    sleep_ms: int = 0
):
    """
    Generate ground truth for queries in input parquet file.
    
    Args:
        input_file: Path to sequenced queries parquet (k50 or k100)
        output_dir: Directory to write neighbors.parquet
        opensearch_url: OpenSearch endpoint
        index: Index name
        top_k: Number of neighbors to retrieve per query
        account_filter: Optional account_id to filter queries (for testing)
        sleep_ms: Milliseconds to sleep between queries (for production)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read queries from parquet
    print(f"Reading queries from: {input_file}")
    table = pq.read_table(input_file)
    data = table.to_pydict()
    
    total_queries = len(data["id"])
    print(f"Total queries in file: {total_queries}")
    print(f"OpenSearch: {opensearch_url}/{index}")
    print(f"Top-k: {top_k}")
    if sleep_ms > 0:
        print(f"Sleep between queries: {sleep_ms}ms")
    
    # Filter by account if specified
    if account_filter:
        print(f"Filtering for account_id: {account_filter}")
        filtered_indices = []
        for i in range(total_queries):
            query_dict = data["query"][i]
            cleaned = remove_nulls(query_dict)
            account_id = extract_account_id(cleaned)
            if account_id == account_filter:
                filtered_indices.append(i)
        print(f"Found {len(filtered_indices)} queries for account {account_filter}")
        indices_to_process = filtered_indices
    else:
        print("Processing ALL queries (no account filter)")
        indices_to_process = list(range(total_queries))
    
    print()
    print("=" * 70)
    print("  GROUND TRUTH GENERATION - EXACT KNN QUERIES")
    print("=" * 70)
    print(f"  Input file:    {input_file}")
    print(f"  Output dir:    {output_dir}")
    print(f"  OpenSearch:    {opensearch_url}/{index}")
    print(f"  Top-k:         {top_k}")
    print(f"  Total queries: {len(indices_to_process)}")
    print(f"  Sleep:         {sleep_ms}ms between queries" if sleep_ms > 0 else "  Sleep:         None (no delay)")
    if account_filter:
        print(f"  Account filter: {account_filter}")
    print("=" * 70)
    print()
    print("Starting sequential query execution...")
    print()
    
    # Process each query ONE BY ONE
    results = []
    errors = []
    total_time = 0
    start_run_time = time.time()
    
    # Track queries with incomplete results
    incomplete_results = []  # List of (id, got_count, expected_count)
    
    for progress_idx, i in enumerate(indices_to_process):
        query_id = data["id"][i]  # Use SAME integer ID from query parquet
        query_dict = data["query"][i]
        
        try:
            # Clean nulls from parquet struct
            cleaned_query = remove_nulls(query_dict)
            
            # Extract account_id for reporting
            account_id = extract_account_id(cleaned_query)
            
            # Convert to exact KNN query
            exact_query = build_exact_knn_query(cleaned_query, top_k)
            
            # Execute query (ONE AT A TIME)
            start_time = time.time()
            neighbors = execute_query(opensearch_url, index, exact_query)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Track incomplete results
            if len(neighbors) < top_k:
                incomplete_results.append({
                    "id": query_id,
                    "account_id": account_id,
                    "got": len(neighbors),
                    "expected": top_k
                })
            
            results.append({
                "id": query_id,  # Same integer ID as query parquet
                "neighbors": neighbors
            })
            
            # Progress update
            processed = progress_idx + 1
            avg_time = total_time / processed
            effective_avg = avg_time + (sleep_ms / 1000) if sleep_ms > 0 else avg_time
            remaining = (len(indices_to_process) - processed) * effective_avg
            remaining_min = remaining / 60
            
            # Status indicator
            status_icon = "✓" if len(neighbors) == top_k else "⚠"
            result_count = f"{len(neighbors)}/{top_k}"
            
            print(f"  {status_icon} [{processed:4d}/{len(indices_to_process)}] "
                  f"id={query_id:<5} | account={account_id or 'N/A':<8} | "
                  f"results={result_count:<7} | time={elapsed:.2f}s | "
                  f"ETA={remaining_min:.1f}min")
            
            # Sleep between queries if specified
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000)
        
        except Exception as e:
            errors.append({"id": query_id, "account_id": account_id if 'account_id' in dir() else 'N/A', "error": str(e)})
            print(f"  ✗ [{progress_idx + 1:4d}/{len(indices_to_process)}] "
                  f"id={query_id:<5} | ERROR: {e}")
    
    # Calculate total elapsed time
    total_run_time = time.time() - start_run_time
    
    # Save to parquet
    if results:
        output_table = pa.Table.from_pydict({
            "id": [r["id"] for r in results],
            "neighbors": [r["neighbors"] for r in results]
        })
        
        output_file = output_dir / "neighbors.parquet"
        pq.write_table(output_table, output_file)
        
        print()
        print("=" * 70)
        print("  GROUND TRUTH GENERATION COMPLETE")
        print("=" * 70)
        print()
        print("  📊 EXECUTION SUMMARY")
        print("  " + "-" * 40)
        print(f"  Queries processed:     {len(results)}")
        print(f"  Successful:            {len(results) - len(errors)}")
        print(f"  Errors:                {len(errors)}")
        print(f"  Incomplete (< k):      {len(incomplete_results)}")
        print()
        print("  ⏱️  TIMING")
        print("  " + "-" * 40)
        print(f"  Total query time:      {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"  Total run time:        {total_run_time:.2f}s ({total_run_time/60:.1f} min)")
        print(f"  Avg time per query:    {total_time / len(results):.2f}s")
        if sleep_ms > 0:
            print(f"  Sleep overhead:        {(sleep_ms * len(results)) / 1000:.2f}s")
        print()
        print("  📁 OUTPUT FILE")
        print("  " + "-" * 40)
        print(f"  Path:     {output_file}")
        print(f"  Size:     {output_file.stat().st_size / 1024:.2f} KB")
        print(f"  Rows:     {len(results)}")
        print()
        print("  📋 SAMPLE DATA (first row)")
        print("  " + "-" * 40)
        print(f"  id:        {results[0]['id']} (integer, matches query parquet)")
        print(f"  neighbors: [{results[0]['neighbors'][0]}, {results[0]['neighbors'][1]}, ...] ({len(results[0]['neighbors'])} total)")
        
        # Report incomplete results
        if incomplete_results:
            print()
            print(f"  ⚠️  INCOMPLETE RESULTS (< {top_k} neighbors)")
            print("  " + "-" * 40)
            print(f"  Count: {len(incomplete_results)} / {len(results)} queries ({100*len(incomplete_results)/len(results):.1f}%)")
            print()
            print(f"  {'ID':<8} {'Account':<12} {'Got':<6} {'Expected':<8} {'Missing':<8}")
            print("  " + "-" * 50)
            for item in incomplete_results:
                missing = item['expected'] - item['got']
                print(f"  {item['id']:<8} {item['account_id'] or 'N/A':<12} {item['got']:<6} {item['expected']:<8} {missing:<8}")
        else:
            print()
            print(f"  ✅ ALL QUERIES RETURNED EXACTLY {top_k} RESULTS")
        
        print()
        print("=" * 70)
    
    if errors:
        print()
        print("  ❌ ERRORS ENCOUNTERED")
        print("  " + "-" * 40)
        for e in errors[:10]:
            print(f"  id={e['id']}, account={e['account_id']}: {e['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth (neighbors.parquet) for custom query dataset"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to sequenced queries parquet (e.g., k50_queries.parquet or k100_queries.parquet)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output directory for neighbors.parquet"
    )
    parser.add_argument(
        "--opensearch-url",
        default="http://localhost:9200",
        help="OpenSearch endpoint (default: http://localhost:9200)"
    )
    parser.add_argument(
        "--index",
        required=True,
        help="OpenSearch index name"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of neighbors to retrieve per query (default: 100)"
    )
    parser.add_argument(
        "--account-filter",
        type=str,
        default=None,
        help="Optional: Filter queries by account_id (for testing with local data)"
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Milliseconds to sleep between queries (default: 0, recommended for prod: 100)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    generate_ground_truth(
        input_file=input_path,
        output_dir=output_path,
        opensearch_url=args.opensearch_url,
        index=args.index,
        top_k=args.top_k,
        account_filter=args.account_filter,
        sleep_ms=args.sleep_ms
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
