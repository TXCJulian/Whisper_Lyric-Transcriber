#!/bin/sh
set -e

PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Adjust appuser UID/GID if needed
if [ "$(id -u appuser)" != "$PUID" ] || [ "$(id -g appuser)" != "$PGID" ]; then
    groupmod -o -g "$PGID" appgroup 2>/dev/null || true
    usermod -o -u "$PUID" -g "$PGID" appuser 2>/dev/null || true
fi

# Ensure ownership of app directories for appuser, even when UID/GID haven't changed
chown -R appuser:appgroup /app/models /app/jobs 2>/dev/null || true

# Grant GPU access: match host render/KFD group GIDs for Intel (/dev/dri) and AMD (/dev/dri, /dev/kfd)
for dev in /dev/dri/renderD* /dev/dri/card* /dev/kfd; do
    [ -e "$dev" ] || continue
    dev_gid=$(stat -c '%g' "$dev")
    if ! id -G appuser | tr ' ' '\n' | grep -q "^${dev_gid}$"; then
        grp_name=$(getent group "$dev_gid" | cut -d: -f1 || true)
        if [ -z "$grp_name" ]; then
            grp_name="devgpu${dev_gid}"
            groupadd -g "$dev_gid" "$grp_name" 2>/dev/null || true
        fi
        usermod -a -G "$grp_name" appuser 2>/dev/null || true
    fi
done

exec gosu appuser "$@"
