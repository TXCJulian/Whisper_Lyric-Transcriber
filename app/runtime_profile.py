import os


def get_runtime_profile() -> str:
    """Return the runtime profile the container started with.

    docker/entrypoint.sh picks the profile (and with it the Python
    environment) before the app starts -- 'nvidia', 'nvidia-legacy',
    'intel', 'amd' or 'cpu' -- and exports it as RUNTIME_PROFILE. Local
    (non-container) runs have no entrypoint, so an explicit
    GPU_BACKEND=nvidia-legacy still selects the legacy profile there;
    anything else returns '' (unknown).
    """
    profile = os.getenv("RUNTIME_PROFILE", "").lower().strip()
    if profile:
        return profile
    if os.getenv("GPU_BACKEND", "").lower().strip() == "nvidia-legacy":
        return "nvidia-legacy"
    return ""
