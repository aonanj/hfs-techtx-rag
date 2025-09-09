#!/usr/bin/env sh
set -e

# choose a canonical persistent path
: "${PERSIST_DIRECTORY:=/data/chroma_db}"
mkdir -p "$PERSIST_DIRECTORY"

# if running as root, fix /data ownership and perms so future runs can write
if [ "$(id -u)" -eq 0 ]; then
  chown -R 1000:1000 /data || true
  chmod -R u+rwX,g+rwX /data || true
fi

exec "$@"
