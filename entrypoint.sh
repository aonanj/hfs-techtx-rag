#!/usr/bin/env sh
set -e

# choose a canonical persistent path
: "${PERSIST_DIRECTORY:=/data/chroma_db}"
mkdir -p "$PERSIST_DIRECTORY"

chown -R 1000:1000 /data || true
chmod -R u+rwX,g+rwX /data || true

exec "$@"
