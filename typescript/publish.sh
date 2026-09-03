#!/usr/bin/env bash
# Publish @openfluke/welvet@1.1.1 (replaces Loom 0.80 on npm).
set -euo pipefail
cd "$(dirname "$0")"
npm run build:all
npm test
npm publish --access public
