#!/usr/bin/env python3
"""
Extract OpenSearch queries from production log files.

Usage:
    python extract_queries_from_logs.py \
        --input /path/to/logfile.json \
        --output /path/to/extracted/ \
        --accounts 340735,301531
"""

import argparse
import json
import re
from pathlib import Path


def extract_query_from_message(message: str) -> tuple[str | None, dict | None]:
    """
    Extract the request UUID and query JSON from a log message.
    
    Returns:
        tuple: (request_uuid, query_dict) or (None, None) if extraction fails
    """
    # Extract UUID from the beginning of the message: [uuid]
    uuid_match = re.match(r'\[([a-f0-9-]+)\]', message)
    request_uuid = uuid_match.group(1) if uuid_match else None
    
    # Find the query={...} part - it starts with "query={" and goes to end of message
    query_match = re.search(r'query=(\{.*)', message, re.DOTALL)
    if not query_match:
        return request_uuid, None
    
    query_str = query_match.group(1).strip()
    
    # The query JSON might have trailing content after the closing brace
    # We need to find the balanced closing brace
    try:
        # Try parsing as-is first
        query_dict = json.loads(query_str)
        return request_uuid, query_dict
    except json.JSONDecodeError:
        # Try to find balanced braces
        brace_count = 0
        end_idx = 0
        for i, char in enumerate(query_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > 0:
            try:
                query_dict = json.loads(query_str[:end_idx])
                return request_uuid, query_dict
            except json.JSONDecodeError:
                return request_uuid, None
    
    return request_uuid, None


def extract_account_id_from_message(message: str) -> str | None:
    """Extract accountId from the log message."""
    match = re.search(r'accountId=(\d+)', message)
    return match.group(1) if match else None


def process_log_file(input_file: Path, output_dir: Path, account_filter: set[str] | None = None):
    """
    Process a log file and extract queries into per-account folders.
    
    Args:
        input_file: Path to the input log file (NDJSON format)
        output_dir: Path to the output directory
        account_filter: Optional set of account IDs to filter (None = all accounts)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"Processing: {input_file}")
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                log_entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: Failed to parse JSON - {e}")
                error_count += 1
                continue
            
            # Get the message field
            message = log_entry.get('_source', {}).get('message', '')
            if not message:
                skipped_count += 1
                continue
            
            # Extract account ID
            account_id = extract_account_id_from_message(message)
            if not account_id:
                skipped_count += 1
                continue
            
            # Filter by account if specified
            if account_filter and account_id not in account_filter:
                skipped_count += 1
                continue
            
            # Extract query
            request_uuid, query_dict = extract_query_from_message(message)
            if not query_dict:
                print(f"  Line {line_num}: Failed to extract query for account {account_id}")
                error_count += 1
                continue
            
            # Create account directory
            account_dir = output_dir / f"account_id={account_id}"
            account_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            if request_uuid:
                filename = f"query_{request_uuid}.json"
            else:
                filename = f"query_{line_num}.json"
            
            # Save query
            query_file = account_dir / filename
            with open(query_file, 'w') as qf:
                json.dump(query_dict, qf, indent=2)
            
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                print(f"  Extracted {extracted_count} queries...")
    
    print(f"\nSummary for {input_file.name}:")
    print(f"  Extracted: {extracted_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="Extract OpenSearch queries from production logs")
    parser.add_argument("--input", required=True, help="Path to input log file (JSON/NDJSON)")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--accounts", help="Comma-separated list of account IDs to filter (optional)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Parse account filter
    account_filter = None
    if args.accounts:
        account_filter = set(args.accounts.split(','))
        print(f"Filtering for accounts: {account_filter}")
    
    # Process the file
    count = process_log_file(input_path, output_path, account_filter)
    
    print(f"\nDone! Total queries extracted: {count}")
    print(f"Output directory: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

