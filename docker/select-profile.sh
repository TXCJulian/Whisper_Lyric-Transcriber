# shellcheck shell=sh
# Sourced by entrypoint.sh: picks the runtime profile -- and with it the
# Python environment -- before the app starts.
#
# The nvidia image ships two environments: /opt/venv/nvidia (cu130 torch +
# faster-whisper) and /opt/venv/nvidia-legacy (cu126 torch + openai-whisper)
# for Maxwell/Pascal/Volta cards, whose kernels cu130 torch and CTranslate2's
# wheels no longer ship. The GPU's compute capability decides, so one image
# tag runs on any NVIDIA generation -- or on CPU when no GPU is visible.
#
# GPU_BACKEND=nvidia/cuda deliberately does NOT force the modern env: older
# configs set it almost everywhere, and a Pascal node must still land on
# the legacy env. Only nvidia-legacy and cpu force a profile.
#
# Exports RUNTIME_PROFILE (read by app/runtime_profile.py) and VENV_DIR, and
# puts $VENV_DIR/bin first on PATH. NVIDIA_SMI and VENV_ROOT are overridable
# for testing.

# cu130 torch needs a CUDA 13 capable host driver; on an older driver the
# modern env would silently see no GPU, while the legacy env (cu126) still runs.
_MIN_DRIVER_CUDA13=580

# Prints the first GPU's "<compute capability>,<driver version>" (e.g.
# "8.6,580.65.06"), or nothing if nvidia-smi is missing, fails, or reports no GPU.
_gpu_query() {
    command -v "${NVIDIA_SMI:-nvidia-smi}" >/dev/null 2>&1 || return 0
    "${NVIDIA_SMI:-nvidia-smi}" --query-gpu=compute_cap,driver_version --format=csv,noheader 2>/dev/null \
        | head -n 1 | tr -d ' \r' || true
}

# _cap_lt A B: true if compute capability A < B (numeric major, then minor).
_cap_lt() {
    _a_major=${1%%.*} _a_minor=${1#*.}
    _b_major=${2%%.*} _b_minor=${2#*.}
    [ "$_a_major" -lt "$_b_major" ] && return 0
    [ "$_a_major" -eq "$_b_major" ] && [ "$_a_minor" -lt "$_b_minor" ]
}

_is_cap() {
    case "$1" in
        [0-9]*.[0-9]*) case "$1" in *[!0-9.]*) return 1 ;; esac; return 0 ;;
        *) return 1 ;;
    esac
}

select_profile() {
    _root=${VENV_ROOT:-/opt/venv}
    _variant=${IMAGE_VARIANT:-nvidia}
    _override=$(printf '%s' "${GPU_BACKEND:-}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

    if [ "$_variant" != "nvidia" ]; then
        RUNTIME_PROFILE=$_variant
        _reason="$_variant image"
    elif [ "$_override" = "nvidia-legacy" ]; then
        RUNTIME_PROFILE=nvidia-legacy
        _reason="forced by GPU_BACKEND"
    elif [ "$_override" = "cpu" ]; then
        RUNTIME_PROFILE=cpu
        _reason="forced by GPU_BACKEND"
    else
        _gpu=$(_gpu_query)
        _cap=${_gpu%%,*}
        _driver_major=""
        case "$_gpu" in *,*) _driver_major=${_gpu#*,}; _driver_major=${_driver_major%%.*} ;; esac
        _min=$(cat "$_root/nvidia.min-compute-cap" 2>/dev/null || true)
        _is_cap "$_min" || _min=7.5
        if ! _is_cap "$_cap"; then
            RUNTIME_PROFILE=cpu
            _reason="no NVIDIA GPU visible"
        elif _cap_lt "$_cap" "$_min"; then
            RUNTIME_PROFILE=nvidia-legacy
            _reason="compute capability $_cap < $_min"
        elif case "$_driver_major" in ''|*[!0-9]*) false ;; esac \
            && [ "$_driver_major" -lt "$_MIN_DRIVER_CUDA13" ]; then
            RUNTIME_PROFILE=nvidia-legacy
            _reason="driver $_driver_major < $_MIN_DRIVER_CUDA13 needed by CUDA 13; update the host driver for faster-whisper"
        else
            RUNTIME_PROFILE=nvidia
            _reason="compute capability $_cap"
        fi
    fi

    # The cpu profile reuses an existing environment: the nvidia image's
    # modern env (CUDA torch runs fine on CPU), or the cpu image's own.
    case "$RUNTIME_PROFILE" in
        nvidia-legacy) VENV_DIR=$_root/nvidia-legacy ;;
        cpu) VENV_DIR=$_root/$_variant ;;
        *) VENV_DIR=$_root/$RUNTIME_PROFILE ;;
    esac

    # CTranslate2's CUDA build dlopen()s libcublas.so.12, while cu130 torch only
    # ships CUDA 13 libraries -- the modern env adds nvidia-cublas-cu12 for it
    # (the old nvidia/cuda:12.x base image provided it system-wide). Only for
    # that env: LD_LIBRARY_PATH outranks RUNPATH, so it must never redirect
    # the legacy env's torch to another CUDA build's libraries.
    if [ "$VENV_DIR" = "$_root/nvidia" ]; then
        for _dir in "$VENV_DIR"/lib/python3*/site-packages/nvidia/cublas/lib; do
            [ -d "$_dir" ] && LD_LIBRARY_PATH=$_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        done
        export LD_LIBRARY_PATH
    fi

    PATH=$VENV_DIR/bin:$PATH
    export RUNTIME_PROFILE VENV_DIR PATH
    echo "runtime profile: $RUNTIME_PROFILE ($_reason)" >&2
}
