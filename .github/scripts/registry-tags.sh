#!/bin/sh
# Usage: registry-tags.sh <image>   (REGISTRY_USER/REGISTRY_TOKEN optional)
#
# Prints the image's tags, one per line. A repository that doesn't exist yet
# prints nothing (first build ever). Any other failure -- registry outage,
# rate limit, auth -- fails, so the build number is never computed from an
# incomplete tag list (that would overwrite immutable X.Y.Z-N tags).
set -eu
image=$1
skopeo=${SKOPEO:-skopeo}
err=$(mktemp)
trap 'rm -f "$err"' EXIT
if [ -n "${REGISTRY_USER:-}" ]; then
    set -- --creds "$REGISTRY_USER:${REGISTRY_TOKEN:-}"
else
    set --
fi
if out=$("$skopeo" list-tags "$@" "docker://$image" 2>"$err"); then
    printf '%s\n' "$out" | jq -r '.Tags // [] | .[]'
elif grep -qiE 'name unknown|repository name not known|manifest unknown|not found' "$err"; then
    echo "registry-tags: $image has no tags yet" >&2
else
    cat "$err" >&2
    exit 1
fi
