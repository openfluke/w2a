#!/usr/bin/env bash
# Publish @openfluke/welvet@1.1.1 to npm (replaces Loom 0.80 line).
#
# Usage:
#   bash publish.sh            # interactive confirm
#   bash publish.sh --dry-run  # build + pack only, no publish
#   bash publish.sh --yes      # skip confirm (CI / you already decided)
#
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

echo "=== @openfluke/welvet — npm publish ==="
echo ""

NAME=$(node -p "require('./package.json').name")
VERSION=$(node -p "require('./package.json').version")
echo "Package: ${NAME}@${VERSION}"
echo ""

if [[ "$VERSION" != "1.1.1" ]]; then
  echo "WARN: package.json version is ${VERSION}, expected 1.1.1 for this Welvet cut." >&2
fi

# 1. Full WASM + TS build
echo "→ npm run build:all"
npm run build:all

if [[ ! -f dist/main.wasm ]]; then
  echo "ERROR: dist/main.wasm missing after build" >&2
  exit 1
fi
if [[ ! -f dist/wasm_exec.js ]]; then
  echo "ERROR: dist/wasm_exec.js missing after build" >&2
  exit 1
fi
if [[ ! -f dist/index.js ]]; then
  echo "ERROR: dist/index.js missing after build" >&2
  exit 1
fi

# Copy browser example into dist for serve / pack curiosity
cp -f examples/browser.html dist/browser.html 2>/dev/null || true

WASM_MB=$(du -m dist/main.wasm | awk '{print $1}')
echo "✓ dist/main.wasm (~${WASM_MB} MB)"
echo ""

# 2. Gate tests (not mega — that is separate)
echo "→ smoke + consumer"
npm run test:smoke
npm run test:consumer

# 3. Live version check against built WASM
node --input-type=module <<'EOF'
import { init, WELVET_ENGINE_VERSION, assertEngineVersion, engineVersion } from "./dist/index.js";
await init();
assertEngineVersion();
const v = engineVersion();
if (v !== WELVET_ENGINE_VERSION) {
  console.error(`version mismatch WASM=${v} package=${WELVET_ENGINE_VERSION}`);
  process.exit(1);
}
console.log(`✓ WASM engineVersion=${v}`);
EOF

echo ""
echo "→ npm pack --dry-run (file list)"
npm pack --dry-run 2>&1 | tail -40

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "Dry run complete — not publishing."
  echo "Tarball preview: npm pack   (creates ${NAME#@}-${VERSION}.tgz with / → - )"
  exit 0
fi

echo ""
if ! npm whoami &>/dev/null; then
  echo "Not logged in to npm."
  echo "  npm login"
  echo "Then re-run: bash publish.sh"
  exit 1
fi
echo "Logged in as: $(npm whoami)"
echo ""
echo "This will PUBLISH ${NAME}@${VERSION} (public) and replace the 0.80 Loom line for new installs of @${VERSION}."
echo "Existing 0.80.0 tag remains on the registry until you deprecate it."
echo ""

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Publish ${NAME}@${VERSION} to npm? [y/N] " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

npm publish --access public --ignore-scripts
echo ""
echo "✓ Published ${NAME}@${VERSION}"
echo "  https://www.npmjs.com/package/${NAME}"
echo ""
echo "Optional — deprecate Loom line:"
echo "  npm deprecate ${NAME}@0.80.0 \"Moved to Welvet ${VERSION}: npm i ${NAME}@${VERSION}\""
