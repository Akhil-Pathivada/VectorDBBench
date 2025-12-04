#!/bin/bash
# Script to run Logstash data extraction in a Podman/Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/Users/akpathivada/Downloads/vdb_dataset}"

echo "Building Logstash container..."
podman build -t logstash-extractor "$SCRIPT_DIR"

echo ""
echo "Starting data extraction..."
echo "  OpenSearch: http://host.docker.internal:9200"
echo "  Output: $OUTPUT_DIR/raw_data"
echo ""

# Run the container
# --network host allows container to reach localhost services
# -v mounts the output directory
podman run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  -v "$OUTPUT_DIR:/data" \
  logstash-extractor

echo ""
echo "Extraction complete! Data saved to: $OUTPUT_DIR/raw_data"

