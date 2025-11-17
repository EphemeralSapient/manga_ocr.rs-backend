# Multi-stage Dockerfile for manga_workflow
# Builds optimized release binary with OpenCV support

# Build stage
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libopencv-dev \
    clang \
    libclang-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY build.rs ./build.rs

# Copy model files (if available)
# Note: In production, fetch these from secure storage
COPY models ./models

# Build release binary
RUN cargo build --release --locked

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-core4.6 \
    libopencv-imgproc4.6 \
    libopencv-imgcodecs4.6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 manga && \
    mkdir -p /app/output /app/fonts && \
    chown -R manga:manga /app

# Copy binary from builder
COPY --from=builder /app/target/release/manga_workflow /usr/local/bin/manga_workflow

# Copy fonts directory
COPY --chown=manga:manga fonts /app/fonts

USER manga
WORKDIR /app

# Expose port for web server
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Run the binary
ENTRYPOINT ["/usr/local/bin/manga_workflow"]
CMD ["--help"]
