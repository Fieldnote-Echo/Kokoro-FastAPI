# Variables for reuse
variable "VERSION" {
    default = "latest"
}

variable "REGISTRY" {
    default = "ghcr.io"
}

variable "OWNER" {
    default = "remsky"
}

variable "REPO" {
    default = "kokoro-fastapi"
}

variable "DOWNLOAD_MODEL" {
    default = "true"
}

# Source-control revision + build timestamp, populated from CI env.
# Left blank for local builds so the resulting labels/annotations stay empty
# rather than carrying stale values.
variable "REVISION" {
    default = ""
}

variable "CREATED" {
    default = ""
}

# OCI metadata applied to every image. `labels` lands in the image config
# (visible via `docker inspect`); `annotations` lands on the pushed manifest
# (which is what GHCR reads for per-arch package pages). Index-level
# annotations for the multi-arch tag are added in release.yml at
# `imagetools create` time, since bake here only produces per-arch manifests.
target "_common" {
    context = "."
    args = {
        DEBIAN_FRONTEND = "noninteractive"
        DOWNLOAD_MODEL = "${DOWNLOAD_MODEL}"
    }
    labels = {
        "org.opencontainers.image.source"   = "https://github.com/${OWNER}/Kokoro-FastAPI"
        "org.opencontainers.image.url"      = "https://github.com/${OWNER}/Kokoro-FastAPI"
        "org.opencontainers.image.licenses" = "Apache-2.0"
        "org.opencontainers.image.revision" = "${REVISION}"
        "org.opencontainers.image.version"  = "${VERSION}"
        "org.opencontainers.image.created"  = "${CREATED}"
    }
    annotations = [
        "org.opencontainers.image.source=https://github.com/${OWNER}/Kokoro-FastAPI",
        "org.opencontainers.image.url=https://github.com/${OWNER}/Kokoro-FastAPI",
        "org.opencontainers.image.licenses=Apache-2.0",
        "org.opencontainers.image.revision=${REVISION}",
        "org.opencontainers.image.version=${VERSION}",
        "org.opencontainers.image.created=${CREATED}",
    ]
}

# Base settings for CPU builds
target "_cpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/cpu/Dockerfile.optimized"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (CPU)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. CPU build."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (CPU)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. CPU build.",
    ]
}

# Base settings for GPU builds
target "_gpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/gpu/Dockerfile.optimized"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (GPU)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. NVIDIA GPU build (CUDA 12.6 amd64 / CUDA 12.9 arm64; cu128 tag for Blackwell)."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (GPU)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. NVIDIA GPU build (CUDA 12.6 amd64 / CUDA 12.9 arm64; cu128 tag for Blackwell).",
    ]
}

# CPU target with multi-platform support
target "cpu" {
    inherits = ["_cpu_base"]
    platforms = ["linux/amd64", "linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}"
    ]
}

# GPU multi-platform: dispatches to per-arch targets so each gets its own CUDA_VERSION
group "gpu" {
    targets = ["gpu-amd64", "gpu-arm64"]
}

# Base settings for AMD ROCm builds
target "_rocm_base" {
    inherits = ["_common"]
    dockerfile = "docker/rocm/Dockerfile"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (ROCm)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. AMD ROCm build (amd64 only)."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (ROCm)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. AMD ROCm build (amd64 only).",
    ]
}


# Individual platform targets for debugging/testing
# CPU base images pinned by digest (multi-arch index digests, so the same pin
# is valid for both amd64 and arm64). Dockerfile tag defaults stay unpinned for
# plain builds; refresh with: docker buildx imagetools inspect python:3.10
target "cpu-amd64" {
    inherits = ["_cpu_base"]
    platforms = ["linux/amd64"]
    args = {
        CPU_BUILDER_IMAGE = "python:3.10@sha256:eeee18553aa04180f626f320c11577b73bf6cfdb77b04894305244eb53c71d50"
        CPU_RUNTIME_IMAGE = "python:3.10-slim@sha256:c1e4e6c01eb489c422288b2de34b0761ca316f7a2d98e2c33f47659a73ed108a"
    }
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}-amd64"
    ]
}

target "cpu-arm64" {
    inherits = ["_cpu_base"]
    platforms = ["linux/arm64"]
    args = {
        CPU_BUILDER_IMAGE = "python:3.10@sha256:eeee18553aa04180f626f320c11577b73bf6cfdb77b04894305244eb53c71d50"
        CPU_RUNTIME_IMAGE = "python:3.10-slim@sha256:c1e4e6c01eb489c422288b2de34b0761ca316f7a2d98e2c33f47659a73ed108a"
    }
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}-arm64"
    ]
}

# CUDA base images pinned by digest (index/manifest-list digests) to prevent
# supply-chain substitution of the mutable version tags. The Dockerfile defaults
# stay unpinned so plain `docker build` / compose builds keep working.
# Refresh with: docker buildx imagetools inspect nvcr.io/nvidia/cuda:<tag>
target "gpu-amd64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/amd64"]
    args = {
        CUDA_VERSION = "12.6.3"
        CUDA_BUILDER_IMAGE = "nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04@sha256:50efab398f76258daa91ceebb33b6467e40217c67ea44fb5a2cebc6be7d9cce3"
        CUDA_RUNTIME_IMAGE = "nvcr.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:8aef630a54bc5c5146ae5ce68e6af5caa3df0fb690bb91544175c91f307e4356"
    }
    # Per-arch tag carries the wheel variant so it parallels gpu-cu128-amd64.
    # The published manifest still resolves to :VERSION / :VERSION-cu126 via release.yml.
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-cu126-amd64"
    ]
}

target "gpu-arm64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/arm64"]
    args = {
        CUDA_VERSION = "12.9.1"
        CUDA_BUILDER_IMAGE = "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04@sha256:a2e1e2360c85298ac47ec2543b406ab1e8cec42e31ee47e4d32140ebc82e1067"
        CUDA_RUNTIME_IMAGE = "nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04@sha256:d02c4310b6d57ca0b16cd80298bdb33a74187baafe2eccd8a6a16180ddc90802"
    }
    # aarch64 uses cu129 wheels (no cu126 aarch64 wheels exist on pytorch.org).
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-cu129-arm64"
    ]
}

# Blackwell / RTX 50-series variant: cu128 torch wheels (sm_120 kernels).
# x86_64 only; published as a -cu128 suffixed tag on the existing -gpu package.
target "gpu-cu128-amd64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/amd64"]
    args = {
        # 12.8.x is the first CUDA toolkit with Blackwell (sm_120) support and is
        # what the cu128 torch wheels are built against. Keep base + wheel aligned.
        CUDA_VERSION = "12.8.1"
        CUDA_BUILDER_IMAGE = "nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed"
        CUDA_RUNTIME_IMAGE = "nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc"
        GPU_EXTRA = "gpu-cu128"
    }
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-cu128-amd64"
    ]
}

# AMD ROCm only supports x86. Base pinned by digest; Dockerfile tag default
# stays unpinned. Refresh: docker buildx imagetools inspect rocm/dev-ubuntu-24.04:6.4.4-complete
target "rocm-amd64" {
    inherits = ["_rocm_base"]
    platforms = ["linux/amd64"]
    args = {
        ROCM_IMAGE = "rocm/dev-ubuntu-24.04:6.4.4-complete@sha256:31418ac10a3769a71eaef330c07280d1d999d7074621339b8f93c484c35f6078"
    }
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-rocm:${VERSION}-amd64"
    ]
}

# Development targets for faster local builds
target "cpu-dev" {
    inherits = ["_cpu_base"]
    # No multi-platform for dev builds
    tags = ["${REGISTRY}/${OWNER}/${REPO}-cpu:dev"]
}

target "gpu-dev" {
    inherits = ["_gpu_base"]
    # No multi-platform for dev builds
    tags = ["${REGISTRY}/${OWNER}/${REPO}-gpu:dev"]
}

target "gpu-cu128-dev" {
    inherits = ["_gpu_base"]
    # No multi-platform for dev builds
    args = {
        CUDA_VERSION = "12.8.1"
        GPU_EXTRA = "gpu-cu128"
    }
    tags = ["${REGISTRY}/${OWNER}/${REPO}-gpu:dev-cu128"]
}

group "dev" {
    targets = ["cpu-dev", "gpu-dev"]
}

# Build groups for different use cases
group "cpu-all" {
    targets = ["cpu", "cpu-amd64", "cpu-arm64"]
}

group "gpu-all" {
    targets = ["gpu-amd64", "gpu-arm64", "gpu-cu128-amd64"]
}

group "rocm-all" {
    targets = ["rocm-amd64"]
}

group "all" {
    targets = ["cpu", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64"]
}

group "individual-platforms" {
    targets = ["cpu-amd64", "cpu-arm64", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64"]
}
