# Logstash Data Extraction Container
# This container runs Logstash to extract data from OpenSearch and write to local disk

FROM docker.elastic.co/logstash/logstash:8.15.3

# Install the OpenSearch input plugin
RUN bin/logstash-plugin install logstash-input-opensearch

# Set working directory
WORKDIR /usr/share/logstash

# Copy the Logstash configuration
COPY logstash_opensearch.conf /usr/share/logstash/pipeline/logstash.conf

# The output directory will be mounted as a volume at runtime
# Default command runs Logstash with the config
CMD ["logstash", "-f", "/usr/share/logstash/pipeline/logstash.conf"]

