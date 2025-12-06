import os
import json
import requests
import time
from pathlib import Path

# Configuration
OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "tickets"
OUTPUT_DIR = "/Users/akpathivada/Downloads/vdb_dataset/raw_data"
MAX_FILE_SIZE = 10240 # 10KB for testing, set to 500 * 1024 * 1024 for prod
SCROLL_TIME = "5m"
BATCH_SIZE = 1000

def get_file_path(account_id, output_dir):
    """Returns the current file path for an account, rotating if needed."""
    account_dir = Path(output_dir) / f"account_id={account_id}"
    account_dir.mkdir(parents=True, exist_ok=True)
    
    # Find latest file or create new
    # Naming: {account_id}-data-{timestamp}.json
    # Simplified rotation: Append counter or check size of current 'active' file
    
    # We will maintain a simple 'current' file pointer in memory if possible, 
    # but for simplicity, let's check the latest file in the dir.
    
    files = sorted(list(account_dir.glob(f"{account_id}-data-*.json")))
    
    current_file = None
    if files:
        current_file = files[-1]
        if current_file.stat().st_size >= MAX_FILE_SIZE:
            current_file = None # Rotate
            
    if not current_file:
        timestamp = int(time.time())
        filename = f"{account_id}-data-{timestamp}.json"
        current_file = account_dir / filename
        
    return current_file

def extract_data():
    print(f"Starting extraction from {OPENSEARCH_URL}/{INDEX_NAME}...")
    
    # Initialize Scroll
    init_url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_search?scroll={SCROLL_TIME}"
    query = {"size": BATCH_SIZE, "query": {"match_all": {}}}
    
    try:
        resp = requests.post(init_url, json=query)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}")
        return

    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]
    total_docs = 0
    
    while hits:
        print(f"Processing batch of {len(hits)} docs...")
        
        # Process batch
        for hit in hits:
            source = hit["_source"]
            account_id = source.get("account_id")
            
            if not account_id:
                continue
                
            # Write to file
            file_path = get_file_path(account_id, OUTPUT_DIR)
            with open(file_path, "a") as f:
                json.dump(source, f)
                f.write("\n")
                
        total_docs += len(hits)
        
        # Get next batch
        scroll_url = f"{OPENSEARCH_URL}/_search/scroll"
        try:
            resp = requests.post(scroll_url, json={"scroll": SCROLL_TIME, "scroll_id": scroll_id})
            resp.raise_for_status()
            data = resp.json()
            hits = data["hits"]["hits"]
            scroll_id = data.get("_scroll_id") # Update scroll ID
        except Exception as e:
            print(f"Error scrolling: {e}")
            break

    print(f"Extraction complete. Total documents: {total_docs}")

if __name__ == "__main__":
    extract_data()

