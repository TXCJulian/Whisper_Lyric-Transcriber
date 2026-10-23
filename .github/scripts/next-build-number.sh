#!/bin/sh
# Usage: <registry tags, one per line> | next-build-number.sh <version> <backend>
#
# Prints the next build number for <version>: one more than the highest N
# among existing "<version>-<N>-<backend>" tags, or 1 if there are none.
# Computed from the registry rather than a CI run counter so it carries over
# unchanged to another CI system.
set -eu
version=$1
backend=$2
# Match dots literally; a bracket class avoids backslash-escaping differences.
escaped=$(printf '%s' "$version" | sed 's/[.]/[.]/g')
max=$(grep -E "^${escaped}-[0-9]+-${backend}\$" \
    | sed -E "s/^${escaped}-([0-9]+)-${backend}\$/\1/" \
    | sort -n | tail -n 1 || true)
echo $(( ${max:-0} + 1 ))
