# Linux build environment for the archives and the checks that link them.
#
# Objects compiled against a newer glibc than the consumer's wheel targets carry
# symbols like __isoc23_strtol that its manylinux policy rejects, so the archives
# are built on the oldest glibc the consumer ships for. The tag pins gcc-toolset
# too; it must not be newer than the consumer's, or objects need libstdc++ parts
# its toolset cannot supply.
ARG ARCH=x86_64
FROM quay.io/pypa/manylinux_2_28_${ARCH}:2026.09.14-1

RUN dnf install -y zlib-devel libcurl-devel openssl-devel && dnf clean all

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path \
    && chmod -R a+rX /opt/rustup /opt/cargo

# Runs as the invoking user, whose home does not exist here. The repository is
# bind-mounted, so git refuses to operate on it as another user.
ENV HOME=/tmp CARGO_HOME=/tmp/cargo
RUN git config --system --add safe.directory '*'

WORKDIR /work
