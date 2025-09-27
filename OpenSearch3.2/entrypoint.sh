#!/bin/bash
# entrypoint.sh — start OpenSearch, wait for REST to be ready, then initialise Security once.
# Notes:
# - We run OpenSearch first (in background), then poll REST (HTTP or HTTPS).
# - We detect whether REST is HTTP or HTTPS on :9200 *inside the container*.
# - We only run securityadmin.sh if the security index is missing (idempotent).
# - If REST is HTTPS, we pass -cacert/-cert/-key; if REST is HTTP, we do not.

set -euo pipefail

# Harden image defaults (not bind-mounted)
chmod 700 /usr/share/opensearch/config || true
chmod 700 /usr/share/opensearch/config/opensearch-performance-analyzer || true
chmod 700 /usr/share/opensearch/config/certs || true

# Hard-fail early if the admin password is not provided (avoids silent auth failures).
: "${OPENSEARCH_INITIAL_ADMIN_PASSWORD:?OPENSEARCH_INITIAL_ADMIN_PASSWORD is required}"

# 1) Start OpenSearch in the background (the official entrypoint sets up and runs the server).
/usr/share/opensearch/opensearch-docker-entrypoint.sh &
OS_PID=$!

# 2) Detect whether REST on 9200 is HTTPS or HTTP (inside the container).
SCHEME="http"
if curl -sSkI https://localhost:9200 >/dev/null 2>&1; then
  SCHEME="https"
fi

echo "⏳ Waiting for OpenSearch to start (${SCHEME^^} on 9200)..."

# Helper: curl with admin auth; adds -k only for HTTPS (self-signed / internal CA).
_curl_auth() {
  if [ "$SCHEME" = "https" ]; then
    curl -sS -k -u "admin:${OPENSEARCH_INITIAL_ADMIN_PASSWORD}" "$@"
  else
    curl -sS    -u "admin:${OPENSEARCH_INITIAL_ADMIN_PASSWORD}" "$@"
  fi
}


# Helper: GET a path on localhost:9200 with the right scheme + auth.
_get() {
  if [ "$SCHEME" = "https" ]; then
    _curl_auth "https://localhost:9200$1"
  else
    _curl_auth "http://localhost:9200$1"
  fi
}

# 3) Wait until REST answers. This prevents initialising Security too early.
until _get "/" >/dev/null 2>&1; do
  sleep 5
done

# 4) Run securityadmin ONLY if the security index is missing/uninitialised (idempotent).
if ! _get "/_cat/indices/.opendistro_security?h=index" | grep -q ".opendistro_security"; then
  echo "Initialising OpenSearch Security from /usr/share/opensearch/config/securityconfig"

  # Decide whether to allow RED cluster during very first bootstrap.
  HEALTH_JSON="$(_get "/_cluster/health" || echo '{}')"
  # Extract the "status" (green/yellow/red), default to "red" if missing.
  HEALTH=$(echo "$HEALTH_JSON" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
  EXTRA_FLAG=""
  if [ "${HEALTH:-red}" = "red" ]; then
    EXTRA_FLAG="--accept-red-cluster"
    echo "Cluster health is RED during bootstrap; passing ${EXTRA_FLAG}"
  fi

  # 5) Run securityadmin with/without TLS client materials depending on REST scheme.
  if [ "$SCHEME" = "https" ]; then
    /usr/share/opensearch/plugins/opensearch-security/tools/securityadmin.sh \
      -cd /usr/share/opensearch/config/securityconfig/ \
      -h localhost -p 9200 \
      -cacert /usr/share/opensearch/config/certs/transport/root-ca.pem \
      -cert   /usr/share/opensearch/config/certs/transport/admin.pem \
      -key    /usr/share/opensearch/config/certs/transport/admin-key.pem \
      -icl -nhnv ${EXTRA_FLAG}
  else
    /usr/share/opensearch/plugins/opensearch-security/tools/securityadmin.sh \
      -cd /usr/share/opensearch/config/securityconfig/ \
      -h localhost -p 9200 \
      -icl -nhnv ${EXTRA_FLAG}
  fi
else
  echo "Security index already present; skipping securityadmin"
fi

# 6) Keep the container in the foreground by waiting for the OpenSearch PID.
echo "OpenSearch REST is ${SCHEME}; waiting on PID ${OS_PID}..."
wait "${OS_PID}"
