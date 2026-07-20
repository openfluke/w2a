#!/usr/bin/env bash
# Auto-release openfluke/w2a for the current Welvet scorecard version.
#
# What it does:
#   1. Read version from ../README.md (welvet scorecard)
#   2. Require logs/suite.txt (+ make suite.pdf via LibreOffice if missing)
#   3. Commit source if dirty (logs/ stay gitignored)
#   4. Push main → origin
#   5. Tag + GitHub Release with suite.txt + suite.pdf as assets
#
# Usage:
#   ./release.sh                 # full release
#   ./release.sh --dry-run       # check assets / version only
#   ./release.sh --no-push       # commit locally, skip push/release
#   ./release.sh --pdf-only      # regenerate suite.pdf from suite.txt, then exit
#
# Needs: git, and either gh (authenticated) or GITHUB_TOKEN.
# Optional: soffice/libreoffice to rebuild suite.pdf from suite.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

WELVET_README="$(cd "$ROOT/.." && pwd)/README.md"
REPO_SLUG="openfluke/w2a"
API="https://api.github.com/repos/${REPO_SLUG}"
LOG_DIR="$ROOT/logs"
SUITE_TXT="$LOG_DIR/suite.txt"
SUITE_PDF="$LOG_DIR/suite.pdf"

DRY_RUN=0
NO_PUSH=0
PDF_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-push) NO_PUSH=1 ;;
    --pdf-only) PDF_ONLY=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

read_version() {
  python3 - <<'PY'
from pathlib import Path
import re
text = Path("../README.md").read_text(encoding="utf-8")
earned = None
m = re.search(r"\*\*(\d+(?:\.\d+)?)\s*/\s*100\*\*\s*pts", text)
if m:
    earned = float(m.group(1))
if earned is None:
    m = re.search(r"\|\s*\*\*Version\*\*\s*\|\s*\*\*(v[\d.]+)\*\*", text)
    if m:
        v = m.group(1)
        earned = 100.0 if v == "v1.0" else float(v[3:]) if v.startswith("v0.") else None
if earned is None:
    raise SystemExit("could not parse Welvet version from ../README.md")
ver = "v1.0" if earned >= 100 else f"v0.{int(round(earned)):02d}"
print(f"{ver} {earned}")
PY
}

have_gh() { command -v gh >/dev/null 2>&1; }

token() {
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo "$GITHUB_TOKEN"
  elif [[ -n "${GH_TOKEN:-}" ]]; then
    echo "$GH_TOKEN"
  else
    echo ""
  fi
}

need_publish_tools() {
  if have_gh; then
    return 0
  fi
  if [[ -n "$(token)" ]]; then
    return 0
  fi
  echo "ERROR: need GitHub CLI (gh) or GITHUB_TOKEN to publish a release." >&2
  echo "  install: https://cli.github.com/  then  gh auth login" >&2
  echo "  or:      export GITHUB_TOKEN=ghp_…" >&2
  exit 1
}

find_soffice() {
  for name in soffice libreoffice; do
    if command -v "$name" >/dev/null 2>&1; then
      command -v "$name"
      return 0
    fi
  done
  return 1
}

ensure_suite_pdf() {
  if [[ -f "$SUITE_PDF" && "$SUITE_PDF" -nt "$SUITE_TXT" ]]; then
    return 0
  fi
  local soffice
  if ! soffice="$(find_soffice)"; then
    if [[ -f "$SUITE_PDF" ]]; then
      echo "  note: LibreOffice not found — keeping existing suite.pdf"
      return 0
    fi
    echo "ERROR: logs/suite.pdf missing and LibreOffice (soffice) not installed." >&2
    echo "  either generate suite.pdf (Writer → Export PDF) or: sudo dnf install libreoffice-writer" >&2
    exit 1
  fi
  echo "→ converting suite.txt → suite.pdf via LibreOffice…"
  mkdir -p "$LOG_DIR"
  # convert into a temp dir then move (LibreOffice can be picky about overwrite)
  local tmp
  tmp="$(mktemp -d)"
  "$soffice" --headless --convert-to pdf --outdir "$tmp" "$SUITE_TXT" >/dev/null
  if [[ ! -f "$tmp/suite.pdf" ]]; then
    echo "ERROR: LibreOffice did not produce suite.pdf" >&2
    rm -rf "$tmp"
    exit 1
  fi
  mv -f "$tmp/suite.pdf" "$SUITE_PDF"
  rm -rf "$tmp"
}

release_exists() {
  local tag="$1"
  if have_gh; then
    gh release view "$tag" --repo "$REPO_SLUG" >/dev/null 2>&1
  else
    local code
    code=$(curl -sS -o /dev/null -w "%{http_code}" \
      -H "Authorization: Bearer $(token)" \
      -H "Accept: application/vnd.github+json" \
      "${API}/releases/tags/${tag}")
    [[ "$code" == "200" ]]
  fi
}

create_or_update_release() {
  local tag="$1"
  local earned="$2"
  local txt_asset="$3"
  local pdf_asset="$4"
  local notes
  notes="$(cat <<EOF
## w2a ${tag}

Welvet validation harness release aligned to scorecard **${earned}/100** → **${tag}**.

### Assets
- \`$(basename "$txt_asset")\` — full suite log (stdout tee)
- \`$(basename "$pdf_asset")\` — printable suite report

### Repo
https://github.com/${REPO_SLUG}

### Regenerate locally
\`\`\`bash
cd welvet/w2a
go run .          # run suites → writes logs/suite.txt
./release.sh      # attach logs + push + GitHub Release
\`\`\`
EOF
)"

  if have_gh; then
    if release_exists "$tag"; then
      echo "  updating existing release ${tag}…"
      gh release upload "$tag" "$txt_asset" "$pdf_asset" --repo "$REPO_SLUG" --clobber
      gh release edit "$tag" --repo "$REPO_SLUG" \
        --title "w2a ${tag}" \
        --notes "$notes"
    else
      echo "  creating release ${tag}…"
      gh release create "$tag" "$txt_asset" "$pdf_asset" --repo "$REPO_SLUG" \
        --title "w2a ${tag}" \
        --notes "$notes"
    fi
    return
  fi

  # curl fallback
  local auth="Authorization: Bearer $(token)"
  local id upload
  if release_exists "$tag"; then
    id=$(curl -sS -H "$auth" -H "Accept: application/vnd.github+json" \
      "${API}/releases/tags/${tag}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  else
    local body
    body=$(VERSION="$tag" NOTES="$notes" python3 - <<'PY'
import json, os
print(json.dumps({
  "tag_name": os.environ["VERSION"],
  "name": f"w2a {os.environ['VERSION']}",
  "body": os.environ["NOTES"],
  "draft": False,
  "prerelease": False,
}))
PY
)
    id=$(curl -sS -H "$auth" -H "Accept: application/vnd.github+json" \
      -H "Content-Type: application/json" \
      -d "$body" "${API}/releases" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  fi

  upload=$(curl -sS -H "$auth" -H "Accept: application/vnd.github+json" \
    "${API}/releases/${id}" | python3 -c "import sys,json; print(json.load(sys.stdin)['upload_url'].split('{')[0])")

  # drop prior assets with same names
  curl -sS -H "$auth" -H "Accept: application/vnd.github+json" \
    "${API}/releases/${id}/assets" | TOKEN="$(token)" python3 - <<'PY'
import json, os, sys, urllib.request
assets = json.load(sys.stdin)
tok = os.environ["TOKEN"]
keep = {os.path.basename(p) for p in sys.argv[1:]} if False else None
PY
  # simpler: delete matching names via python one-liner
  for asset in "$txt_asset" "$pdf_asset"; do
    name="$(basename "$asset")"
    curl -sS -H "$auth" -H "Accept: application/vnd.github+json" \
      "${API}/releases/${id}/assets" \
      | NAME="$name" TOKEN="$(token)" API="$API" python3 -c '
import json,os,sys,urllib.request
name=os.environ["NAME"]; tok=os.environ["TOKEN"]; api=os.environ["API"]
for a in json.load(sys.stdin):
    if a.get("name")==name:
        req=urllib.request.Request(api+"/releases/assets/"+str(a["id"]), method="DELETE",
            headers={"Authorization":"Bearer "+tok,"Accept":"application/vnd.github+json"})
        try: urllib.request.urlopen(req)
        except Exception: pass
'
    ctype="text/plain"
    [[ "$asset" == *.pdf ]] && ctype="application/pdf"
    echo "  uploading $name…"
    curl -sS -H "$auth" -H "Content-Type: $ctype" \
      --data-binary @"$asset" \
      "${upload}?name=${name}" >/dev/null
  done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ ! -f "$WELVET_README" ]]; then
  echo "ERROR: welvet README not found at $WELVET_README" >&2
  exit 1
fi

read -r VERSION EARNED <<<"$(read_version)"
TXT_ASSET_NAME="w2a-suite-${VERSION}.txt"
PDF_ASSET_NAME="w2a-suite-${VERSION}.pdf"
DIST="$ROOT/dist"
mkdir -p "$DIST"
TXT_ASSET="$DIST/$TXT_ASSET_NAME"
PDF_ASSET="$DIST/$PDF_ASSET_NAME"

echo "════════════════════════════════════════"
echo " w2a release"
echo " version:  ${VERSION}  (${EARNED}/100)"
echo " repo:     ${REPO_SLUG}"
echo " assets:   logs/suite.txt + logs/suite.pdf"
echo "════════════════════════════════════════"

if [[ ! -f "$SUITE_TXT" ]]; then
  echo "ERROR: missing $SUITE_TXT" >&2
  echo "  run suites first:  cd welvet/w2a && go run ." >&2
  exit 1
fi

ensure_suite_pdf

if [[ ! -f "$SUITE_PDF" ]]; then
  echo "ERROR: missing $SUITE_PDF" >&2
  exit 1
fi

# versioned copies for the release (logs/ names stay stable)
cp -f "$SUITE_TXT" "$TXT_ASSET"
cp -f "$SUITE_PDF" "$PDF_ASSET"

echo "→ suite.txt  $(du -h "$SUITE_TXT" | cut -f1)"
echo "→ suite.pdf  $(du -h "$SUITE_PDF" | cut -f1)  →  ${PDF_ASSET_NAME}"

if [[ "$PDF_ONLY" -eq 1 ]]; then
  echo "→ --pdf-only: done"
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "dry-run: skipping commit / push / release"
  echo "would release: ${VERSION}"
  echo "  - ${TXT_ASSET_NAME}"
  echo "  - ${PDF_ASSET_NAME}"
  exit 0
fi

# Commit source (never commit logs/ or dist/)
echo ""
echo "→ git status"
git status --short || true

if [[ -n "$(git status --porcelain)" ]]; then
  echo "→ committing source…"
  git add -A
  git reset -q -- logs/ dist/ 2>/dev/null || true
  # re-check — maybe only logs changed
  if [[ -n "$(git diff --cached --name-only)" ]]; then
    git commit -m "$(cat <<EOF
Release w2a ${VERSION}

Aligned to Welvet scorecard ${EARNED}/100.
Suite log artifacts published on GitHub Release (not committed).
EOF
)"
  else
    echo "→ nothing staged (logs/dist only) — skip commit"
  fi
else
  echo "→ working tree clean — nothing to commit"
fi

if [[ "$NO_PUSH" -eq 1 ]]; then
  echo "→ --no-push: skipping push + GitHub release"
  echo "  assets ready: $TXT_ASSET  $PDF_ASSET"
  exit 0
fi

need_publish_tools

echo "→ pushing main…"
git push origin HEAD

echo "→ publishing GitHub Release ${VERSION}…"
if git rev-parse "$VERSION" >/dev/null 2>&1; then
  echo "  tag ${VERSION} already exists locally"
else
  git tag -a "$VERSION" -m "w2a ${VERSION} (Welvet scorecard ${EARNED}/100)"
fi
git push origin "$VERSION" 2>/dev/null || git push origin "refs/tags/${VERSION}"

create_or_update_release "$VERSION" "$EARNED" "$TXT_ASSET" "$PDF_ASSET"

echo ""
echo "════════════════════════════════════════"
echo " Done · ${VERSION}"
echo " Repo:    https://github.com/${REPO_SLUG}"
echo " Release: https://github.com/${REPO_SLUG}/releases/tag/${VERSION}"
echo " TXT:     https://github.com/${REPO_SLUG}/releases/latest/download/${TXT_ASSET_NAME}"
echo " PDF:     https://github.com/${REPO_SLUG}/releases/latest/download/${PDF_ASSET_NAME}"
echo "════════════════════════════════════════"
