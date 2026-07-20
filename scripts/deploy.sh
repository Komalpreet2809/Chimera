#!/usr/bin/env bash
# Chimera one-shot deploy: backend -> Hugging Face Spaces, frontend -> Vercel.
#
# Prerequisite (one time):   hf auth login
#
# Usage:                     bash scripts/deploy.sh
#
# The order matters. The frontend inlines NEXT_PUBLIC_API_BASE at BUILD time,
# so the backend URL has to exist and be registered with Vercel before the
# frontend is built — otherwise the deployed site ships pointing at localhost.

set -euo pipefail

SPACE_NAME="${SPACE_NAME:-chimera}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "Checking credentials"

HF_USER="$(python -c "
from huggingface_hub import whoami
try:
    print(whoami()['name'])
except Exception:
    pass
" 2>/dev/null || true)"

[ -n "$HF_USER" ] || die "Not logged in to Hugging Face. Run:  hf auth login"
echo "    Hugging Face: $HF_USER"

vercel whoami >/dev/null 2>&1 || die "Not logged in to Vercel. Run:  vercel login"
echo "    Vercel:       $(vercel whoami 2>/dev/null | tail -1)"

SPACE_ID="${HF_USER}/${SPACE_NAME}"
SPACE_URL="https://${HF_USER//./-}-${SPACE_NAME}.hf.space"

# ------------------------------------------------------------ backend first
say "Creating Space $SPACE_ID (if it doesn't exist)"
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$SPACE_ID', repo_type='space', space_sdk='docker', exist_ok=True)
print('    ok')
"

say "Uploading backend to $SPACE_ID"
# README_SPACE.md carries the YAML front-matter the Space needs, so it is
# uploaded as README.md. Weights are never uploaded — the image pulls them.
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id='$SPACE_ID',
    repo_type='space',
    folder_path='backend',
    ignore_patterns=['**/__pycache__/**','**/*.pyc','tests/**','scripts/**',
                     '.venv/**','**/.pytest_cache/**'],
)
api.upload_file(
    path_or_fileobj='backend/README_SPACE.md',
    path_in_repo='README.md',
    repo_id='$SPACE_ID',
    repo_type='space',
)
print('    ok')
"

echo "    Space building at: https://huggingface.co/spaces/$SPACE_ID"
echo "    It will serve at:  $SPACE_URL"

say "Waiting for the Space to come up (first build pulls torch + weights, ~5-10 min)"
for i in $(seq 1 120); do
  if curl -fsS -m 10 "$SPACE_URL/api/health" >/dev/null 2>&1; then
    echo "    healthy: $(curl -fsS -m 10 "$SPACE_URL/api/health")"
    break
  fi
  [ "$i" -eq 120 ] && die "Space did not become healthy. Check the build logs at https://huggingface.co/spaces/$SPACE_ID"
  printf '.'
  sleep 15
done

# --------------------------------------------------------------- frontend
say "Pointing the frontend at $SPACE_URL"
cd "$ROOT/frontend"
# Replace rather than append, so re-running doesn't stack duplicates.
vercel env rm NEXT_PUBLIC_API_BASE production --yes >/dev/null 2>&1 || true
printf '%s' "$SPACE_URL" | vercel env add NEXT_PUBLIC_API_BASE production >/dev/null
echo "    set"

say "Deploying frontend to Vercel"
vercel deploy --prod --yes

say "Done"
cat <<EOF

  Backend   $SPACE_URL
  Frontend  https://chimera.komalpreet.me

  If the custom domain doesn't resolve yet, add this DNS record in Cloudflare:

      Type   A
      Name   chimera
      Value  76.76.21.21
      Proxy  DNS only  (grey cloud, NOT orange)

  The proxy must be off — Cloudflare's proxy in front of Vercel breaks
  certificate issuance for the domain.

EOF
