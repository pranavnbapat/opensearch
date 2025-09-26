#!/bin/bash
set -euo pipefail

/usr/share/opensearch/opensearch-docker-entrypoint.sh &

SCHEME="http"
if curl -sSkI https://localhost:9200 >/dev/null 2>&1; then
  SCHEME="https"
fi

echo "⏳ Waiting for OpenSearch to start (HTTPS on 9200)..."
until curl -sk -u "admin:${OPENSEARCH_INITIAL_ADMIN_PASSWORD}" https://localhost:9200 >/dev/null; do
  sleep 5
done

# Helper: GET a URL with or without -k depending on scheme, no auth
_curl_head() {
  if [ "$SCHEME" = "https" ]; then
    curl -sS -k -I "https://localhost:9200$1"
  else
    curl -sS    -I "http://localhost:9200$1"
  fi
}

_curl_get() {
  if [ "$SCHEME" = "https" ]; then
    curl -sS -k    "https://localhost:9200$1"
  else
    curl -sS       "http://localhost:9200$1"
  fi
}

# Initialise/upgrade the security index using your transport CA + admin cert
#/usr/share/opensearch/plugins/opensearch-security/tools/securityadmin.sh \
#  -cd /usr/share/opensearch/config/securityconfig/ \
#  -cacert /usr/share/opensearch/config/certs/transport/root-ca.pem \
#  -cert /usr/share/opensearch/config/certs/transport/admin.pem \
#  -key  /usr/share/opensearch/config/certs/transport/admin-key.pem \
#  -icl -nhnv --accept-red-cluster

# --- Run securityadmin ONLY if security index is missing/uninitialised ---
if ! _curl_get "/_cat/indices/.opendistro_security?h=index" | grep -q ".opendistro_security"; then
  echo "🔐 Initialising OpenSearch Security from /usr/share/opensearch/config/securityconfig"

  # Decide whether to allow RED cluster during very first bootstrap
  HEALTH_JSON=$(_curl_get "/_cluster/health" || echo '{}')
  # Pull .status or default to "red" if missing
  HEALTH=$(echo "$HEALTH_JSON" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
  EXTRA_FLAG=""
  if [ "${HEALTH:-red}" = "red" ]; then
    EXTRA_FLAG="--accept-red-cluster"
    echo "⚠  Cluster health is RED during bootstrap; passing ${EXTRA_FLAG}"
  fi

  # securityadmin: pass host/port explicitly; HTTPS toggled by -cacert/-cert/-key
  /usr/share/opensearch/plugins/opensearch-security/tools/securityadmin.sh \
    -cd /usr/share/opensearch/config/securityconfig/ \
    -h localhost \
    -p 9200 \
    -cacert /usr/share/opensearch/config/certs/transport/root-ca.pem \
    -cert   /usr/share/opensearch/config/certs/transport/admin.pem \
    -key    /usr/share/opensearch/config/certs/transport/admin-key.pem \
    -icl -nhnv ${EXTRA_FLAG}
else
  echo "✅ Security index already present; skipping securityadmin"
fi

# Run securityadmin ONLY if security index is missing/uninitialised
#if ! curl -sk -u "admin:${OPENSEARCH_INITIAL_ADMIN_PASSWORD}" \
#  https://localhost:9200/_cat/indices/.opendistro_security?h=index >/dev/null; then
#  echo "Initialising OpenSearch Security from /usr/share/opensearch/config/securityconfig"
#  /usr/share/opensearch/plugins/opensearch-security/tools/securityadmin.sh \
#    -cd /usr/share/opensearch/config/securityconfig/ \
#    -icl -nhnv --accept-red-cluster \
#    -cacert /usr/share/opensearch/config/certs/transport/root-ca.pem \
#    -cert   /usr/share/opensearch/config/certs/transport/admin.pem \
#    -key    /usr/share/opensearch/config/certs/transport/admin-key.pem
#else
#  echo "Security index already present; skipping securityadmin"
#fi

wait
