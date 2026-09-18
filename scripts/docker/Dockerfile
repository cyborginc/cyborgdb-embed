# Linux build environment for the ONNX Runtime static build and symbol audit.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      python3 \
      git \
      ca-certificates \
      binutils \
      file \
      libcurl4-openssl-dev \
      libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# The repository is bind-mounted, so git refuses to operate on it as another user.
RUN git config --global --add safe.directory '*'

WORKDIR /work
