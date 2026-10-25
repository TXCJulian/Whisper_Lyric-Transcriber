# Kubernetes examples

Generic [Kustomize](https://kustomize.io/) manifests: a starting point you adapt
to your own cluster (namespace, image tag, storage class, ingress).

```
deploy/k8s/
  base/              Deployment, Service, PVC (no GPU request: runs on any node)
  overlays/nvidia/   base + one NVIDIA GPU (runtimeClassName: nvidia, nvidia.com/gpu: 1)
```

## Usage

```bash
# 1. Secret with your Genius token (HF_TOKEN is optional)
kubectl create secret generic lyric-transcriber \
  --from-literal=GENIUS_ACCESS_TOKEN=<token>

# 2a. With an NVIDIA GPU (needs the NVIDIA device plugin / GPU operator)
kubectl apply -k deploy/k8s/overlays/nvidia

# 2b. CPU only
kubectl apply -k deploy/k8s/base
```

Pin the image to an immutable `<version>-<build>-<variant>` tag (e.g.
`2.0.0-1-nvidia`) rather than `latest-nvidia`, so a rollout is always a deliberate
change. The `nvidia` image runs on every NVIDIA generation (it picks the modern or
the Maxwell/Pascal/Volta environment from the GPU it gets) and on CPU when no GPU is
visible. Check what a pod chose with `GET /health` (`runtime_profile`).

## GPU → CPU failover

Kubernetes can't express "use a GPU if the node has one, otherwise run on CPU" in a
single pod spec: a pod that requests `nvidia.com/gpu` can only be scheduled on GPU
nodes. Two common ways around it:

1. **No GPU request, NVIDIA runtime as the node default.** Deploy `base` (no GPU
   request). On GPU nodes, make `nvidia` containerd's default runtime; the image sets
   `NVIDIA_VISIBLE_DEVICES=all`, so the GPU becomes visible without a request and the
   pod uses it. On CPU-only nodes the same pod falls back to CPU. Trade-off: the
   scheduler doesn't account for the GPU, so nothing stops another GPU workload from
   sharing it.
2. **Two Deployments.** The `nvidia` overlay as the primary, plus `base` (with a
   different name) scaled to 0, and something — you, an operator, or your
   automation — scales the CPU one up when the GPU node is gone.

Either way, failover between nodes needs the model cache on storage every node can
reach: replace the ReadWriteOnce PVC with a ReadWriteMany storage class (e.g. NFS), or
accept a fresh ~5 GB model download on the new node.

## Batch processing

To let `POST /batch` read songs from your library, mount it into the pod and point
`TRANSCRIBER_ALLOWED_ROOTS` at the mount path — commented examples for the volume, the
mount and the env vars are in `base/deployment.yaml`. A read-only mount is enough for the
default `TRANSCRIBER_BATCH_OUTPUT=download`; `alongside` writes lyrics next to the songs
and needs a writable volume. Batches live in the pod's memory, like jobs: a restart or a
failover to another node loses running batches, so the client should restart them.
