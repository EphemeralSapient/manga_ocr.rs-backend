# syntax=docker/dockerfile:1.4

# ============================================================================
# Build Stage
# ============================================================================
FROM rust:1.83-slim-bookworm AS builder

# Install build dependencies in a single layer
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libopencv-dev \
    clang \
    libclang-dev \
    cmake

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY Cargo.toml Cargo.lock build.rs ./

# Create dummy source to cache dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release --locked && \
    rm -rf src target/release/deps/manga_workflow*

# Copy actual source code
COPY src ./src

# Copy models (required for binary)
COPY models ./models

# Build release binary with cache mounts
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --locked && \
    cp target/release/manga_workflow /app/manga_workflow

# ============================================================================
# Runtime Stage
# ============================================================================
FROM debian:bookworm-slim

# OCI labels for compliance
LABEL org.opencontainers.image.title="Manga Text Processor" \
      org.opencontainers.image.description="High-performance Rust backend for manga text detection, translation, and rendering" \
      org.opencontainers.image.vendor="manga_workflow" \
      org.opencontainers.image.licenses="See LICENSE file" \
      org.opencontainers.image.source="https://github.com/yourusername/manga_workflow"

# Install minimal runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.6 \
    libopencv-imgproc4.6 \
    libopencv-imgcodecs4.6 \
    ca-certificates \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash manga && \
    mkdir -p /app/.cache /app/fonts && \
    chown -R manga:manga /app

WORKDIR /app

# Copy binary and required files
COPY --from=builder --chown=manga:manga /app/manga_workflow /usr/local/bin/manga_workflow
COPY --chown=manga:manga fonts ./fonts
COPY --chown=manga:manga models ./models

# Switch to non-root user
USER manga

# Expose default server port
EXPOSE 1420

# Health check using wget (more minimal than curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:1420/health || exit 1

# Environment variables
ENV SERVER_HOST=0.0.0.0 \
    SERVER_PORT=1420 \
    LOG_LEVEL=INFO \
    RUST_BACKTRACE=1

# Run the server
ENTRYPOINT ["/usr/local/bin/manga_workflow"]
