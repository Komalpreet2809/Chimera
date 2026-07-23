#!/usr/bin/env bash
# Chimera deploy: backend -> Google Cloud Run (free tier), frontend -> Vercel.
#
# Prerequisites (one time):
#   1. Install gcloud:  https://cloud.google.com/sdk/docs/install
#   2. gcloud auth login
#   3. A GCP project with billing enabled (free tier still applies; the card is
#      only for identity verification). Note its PROJECT ID.
#
# Usage:
#   PROJECT_ID=your-project-id bash scripts/deploy-cloudrun.sh
#
# Order matters: the frontend inlines NEXT_PUBLIC_API_BASE at BUILD time, so the
# backend URL must exist before the frontend is built.

set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-chimera-backend}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "Checking prerequisites"
command -v gcloud >/dev/null 2>&1 || die "gcloud not installed — https://cloud.google.com/sdk/docs/install"
command -v vercel >/dev/null 2>&1 || die "vercel CLI not installed"

ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
[ -n "$ACCOUNT" ] && [ "$ACCOUNT" != "(unset)" ] || die "Not logged in — run:  gcloud auth login"
echo "    gcloud account: $ACCOUNT"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
[ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ] || die "No project. Run with:  PROJECT_ID=your-id bash scripts/deploy-cloudrun.sh"
echo "    project:        $PROJECT_ID"
echo "    region:         $REGION"

gcloud config set project "$PROJECT_ID" >/dev/null

say "Enabling required APIs (first run only, ~1 min)"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com >/dev/null
echo "    ok"

# ------------------------------------------------------------ backend first
say "Building and deploying the backend to Cloud Run"
echo "    (first build installs torch + bakes in GPT-2 weights — a few minutes)"
gcloud run deploy "$SERVICE" \
  --source backend \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 20 \
  --max-instances 3 \
  --port 8080

SPACE_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
[ -n "$SPACE_URL" ] || die "Could not read the Cloud Run URL"
echo "    backend live at: $SPACE_URL"

say "Waiting for the service to answer /api/health"
for i in $(seq 1 40); do
  if curl -fsS -m 20 "$SPACE_URL/api/health" >/dev/null 2>&1; then
    echo "    healthy: $(curl -fsS -m 20 "$SPACE_URL/api/health")"
    break
  fi
  [ "$i" -eq 40 ] && die "Service did not become healthy — check logs:  gcloud run services logs read $SERVICE --region $REGION"
  printf '.'
  sleep 6
done

# --------------------------------------------------------------- frontend
say "Pointing the frontend at $SPACE_URL"
cd "$ROOT/frontend"
vercel env rm NEXT_PUBLIC_API_BASE production --yes >/dev/null 2>&1 || true
printf '%s' "$SPACE_URL" | vercel env add NEXT_PUBLIC_API_BASE production >/dev/null
echo "    set"

say "Deploying frontend to Vercel (production)"
vercel deploy --prod --yes

say "Done"
cat <<EOF

  Backend   $SPACE_URL
  Frontend  https://chimera.komalpreet.me

  LAST STEP — the custom domain needs one Cloudflare DNS record:

      Type   A
      Name   chimera
      Value  76.76.21.21
      Proxy  DNS only   (grey cloud, NOT orange)

  Until that record exists, use the *.vercel.app URL that the deploy above
  printed. The proxy must be off or Vercel can't issue the certificate.

EOF
