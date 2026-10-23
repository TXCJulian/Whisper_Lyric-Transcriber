#!/bin/sh
# Usage: <registry tags, one per line> | next-build-number.sh <version>
#
# Prints the next build number for <version>: one more than the highest N
# among existing "<version>-<N>-<variant>" tags of ANY variant, or 1 if there
# are none. Taking the max across variants means a run where one variant
# failed (or was cancelled) never makes the next run reuse an N another
# variant already pushed -- X.Y.Z-N tags must stay immutable. Computed from
# the registry rather than a CI run counter so it carries over unchanged to
# another CI system.
set -eu
version=$1
# Match dots literally; a bracket class avoids backslash-escaping differences.
escaped=$(printf '%s' "$version" | sed 's/[.]/[.]/g')
max=$(grep -E "^${escaped}-[0-9]+-[a-z][a-z0-9-]*\$" \
    | sed -E "s/^${escaped}-([0-9]+)-.*\$/\1/" \
    | sort -n | tail -n 1 || true)
echo $(( ${max:-0} + 1 ))
