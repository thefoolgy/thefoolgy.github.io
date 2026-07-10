#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if rg -n 'https?://localhost:1313|livereload\.js' \
  --glob '*.html' \
  --glob '*.xml' \
  "$repo_root"; then
  echo "Published files must not contain Hugo development-server URLs or livereload scripts." >&2
  exit 1
fi

echo "Public links do not reference the Hugo development server."
