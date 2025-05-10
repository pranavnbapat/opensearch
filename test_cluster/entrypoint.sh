#!/bin/bash
set -e

# Start OpenSearch in background
/usr/share/opensearch/opensearch-docker-entrypoint.sh &

# Wait until OpenSearch is up
echo "⏳ Waiting for OpenSearch to start..."
until curl -k --silent https://localhost:9200 -u admin:${OPENSEARCHTEST_ADMIN_PASSWORD} >/dev/null; do
  sleep 5
done

# Wait for .opendistro_security index to exist
echo "⏳ Waiting for .opendistro_security index to be created..."
until curl -k -u admin:${OPENSEARCHTEST_ADMIN_PASSWORD} \
  -XGET "https://localhost:9200/_cat/indices/.opendistro_security?h=status" 2>/dev/null | grep -qE 'green|yellow'; do
  sleep 5
done

# Wait for .opendistro_security to be green (ensure it's assigned and started)
echo "⏳ Waiting for .opendistro_security to be green..."
until curl -k -u admin:${OPENSEARCHTEST_ADMIN_PASSWORD} \
  -XGET "https://localhost:9200/_cluster/health/.opendistro_security?wait_for_status=green&timeout=1s" 2>/dev/null | grep -q '"status":"green"'; do
  sleep 5
done

echo "✅ .opendistro_security index is green. Running securityadmin.sh..."

# Run securityadmin.sh
/usr/share/opensearch/plugins/opensearch-security/tools/securityadmin.sh \
  -cd /usr/share/opensearch/config/securityconfig/ \
  -icl \
  -nhnv \
  -cacert /usr/share/opensearch/config/certs/fullchain.pem \
  -cert /usr/share/opensearch/config/certs/fullchain.pem \
  -key /usr/share/opensearch/config/certs/privkey-pkcs8.pem

# Wait for OpenSearch to stay running
wait
